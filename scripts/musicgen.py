"""Generate one original, license-free music track PER CLIP, sized to the clip's exact length.

usage: python musicgen.py CLIPS.json MUSIC_DIR
Writes MUSIC_DIR/NN_<clip-name>.wav (44.1 kHz stereo 16-bit).

Each clip may carry  "music": {"style": "epic|driving|tense|uplift|chill|trailer",
                               "key": "A"|"F#"|..., "seed": 1}
Anything missing is chosen automatically (adjacent clips never share a style).
Tempo is nudged so an integer number of bars fits the clip exactly; the track
builds through an intensity arc and ends on a downbeat hit at the clip's end.
Requires numpy + scipy.
"""
import hashlib, json, os, sys, wave
import numpy as np
from scipy import signal

SR = 44100
NOTES = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
         'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
MINOR, MAJOR = [0, 2, 3, 5, 7, 8, 10], [0, 2, 4, 5, 7, 9, 11]

STYLES = {
    'epic':    dict(bpm=76,  mode='minor', progs=[[0, 5, 2, 6], [0, 5, 3, 4], [0, 2, 5, 6]], keys=['D', 'E', 'C#'],
                    pad_cut=1800, pad_att=0.9, bass='whole', drums='cinematic', arp=None, lead='saw',
                    verb=0.5, curve='build', sc=0.0, riser=True, stabs=False),
    'driving': dict(bpm=126, mode='minor', progs=[[0, 5, 2, 6], [0, 3, 5, 4], [0, 0, 6, 5]], keys=['A', 'F#', 'G'],
                    pad_cut=2800, pad_att=0.05, bass='offbeat', drums='four', arp='pluck16', lead='pluck',
                    verb=0.2, curve='wave', sc=0.65, riser=True, stabs=False),
    'tense':   dict(bpm=92,  mode='minor', progs=[[0, 0, 5, 5], [0, 6, 0, 1], [0, 5, 0, 6]], keys=['C', 'F', 'B'],
                    pad_cut=900, pad_att=1.2, bass='pulse', drums='sparse', arp='tick', lead=None,
                    verb=0.5, curve='build', sc=0.0, riser=True, stabs=False, dissonance=True),
    'uplift':  dict(bpm=112, mode='major', progs=[[0, 4, 5, 3], [0, 3, 4, 4], [0, 5, 3, 4]], keys=['G', 'D', 'A'],
                    pad_cut=3500, pad_att=0.15, bass='eighths', drums='pop', arp='pluck8', lead='pluck',
                    verb=0.3, curve='wave', sc=0.3, riser=True, stabs=False),
    'chill':   dict(bpm=88,  mode='minor', progs=[[1, 4, 0, 0], [0, 3, 6, 2], [5, 6, 0, 0]], keys=['F', 'Bb', 'Eb'],
                    pad_cut=1500, pad_att=0.4, bass='syncop', drums='lofi', arp=None, lead='ep',
                    verb=0.3, curve='plateau', sc=0.0, riser=False, stabs=False, keys_ep=True, ext7=True, vinyl=True),
    'trailer': dict(bpm=100, mode='minor', progs=[[0, 0, 5, 6], [0, 6, 5, 6], [0, 2, 5, 4]], keys=['D', 'F#', 'A'],
                    pad_cut=1500, pad_att=0.5, bass='whole', drums='halftime', arp=None, lead=None,
                    verb=0.35, curve='build', sc=0.0, riser=True, stabs=True),
}
ORDER = list(STYLES)


# ---------- DSP helpers ----------
def hz(m): return 440.0 * 2 ** ((m - 69) / 12)
def sos_f(x, kind, f, order=2): return signal.sosfilt(signal.butter(order, f, kind, fs=SR, output='sos'), x)
def tt(n): return np.arange(n) / SR


def saw(f, n, ph=0.0): return 2 * ((f * tt(n) + ph) % 1.0) - 1


def supersaw(f, n, rng, det=0.007, voices=5):
    out = np.zeros(n)
    for k in range(voices):
        out += saw(f * (1 + det * (k - (voices - 1) / 2) / ((voices - 1) / 2 or 1)), n, rng.random())
    return out / voices


def adsr(n, a, d, s, r):
    e = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na, nr = min(na, n), min(nr, n)
    if na: e[:na] = np.linspace(0, 1, na)
    nd = min(nd, max(0, n - na))
    if nd: e[na:na + nd] = np.linspace(1, s, nd)
    if nr: e[n - nr:] *= np.linspace(1, 0, nr)
    return e


def pan_gains(p): return np.cos(p * np.pi / 2), np.sin(p * np.pi / 2)


class Bus:
    def __init__(self, n): self.a = np.zeros((2, n)); self.n = n

    def add(self, start, sig, pan=0.5, gain=1.0):
        if start >= self.n or start + len(sig) <= 0: return
        s0 = max(0, -start); e = min(len(sig), self.n - start)
        gl, gr = pan_gains(pan)
        self.a[0, start + s0:start + e] += sig[s0:e] * gl * gain
        self.a[1, start + s0:start + e] += sig[s0:e] * gr * gain


# ---------- drum / fx one-shots ----------
def kick(rng):
    t = tt(int(0.45 * SR)); f = 42 + 120 * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 1.0 + np.exp(-t * 400) * 0.3


def snare(rng):
    t = tt(int(0.3 * SR)); nz = sos_f(rng.standard_normal(len(t)), 'high', 1500)
    return (nz * np.exp(-t * 18) * 0.6 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) * 0.4)


def clap(rng):
    t = tt(int(0.25 * SR)); nz = sos_f(rng.standard_normal(len(t)), 'band', [900, 3500])
    env = np.exp(-t * 22) + 0.6 * np.exp(-((t - 0.012) ** 2) * 9000) + 0.5 * np.exp(-((t - 0.024) ** 2) * 9000)
    return nz * env * 0.8


def hat(rng, open_=False):
    t = tt(int((0.35 if open_ else 0.06) * SR))
    return sos_f(rng.standard_normal(len(t)), 'high', 7000) * np.exp(-t * (14 if open_ else 90)) * 0.35


def tom(rng, f0=95):
    t = tt(int(0.6 * SR)); f = f0 * (1 + 0.8 * np.exp(-t * 18))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6)


def boom(rng, length=3.0):
    t = tt(int(length * SR)); f = 34 + 70 * np.exp(-t * 6)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6) +
            sos_f(rng.standard_normal(len(t)), 'low', 500) * np.exp(-t * 3) * 0.4)


def crash(rng, length=2.5):
    t = tt(int(length * SR)); return sos_f(rng.standard_normal(len(t)), 'high', 3500) * np.exp(-t * 2.2) * 0.45


def riser(rng, length):
    n = int(length * SR); t = tt(n)
    nz = sos_f(rng.standard_normal(n), 'band', [1500, 9000])
    return nz * (t / length) ** 2.2 * 0.5


def ep_note(f, n):  # FM electric piano
    t = tt(n); return np.sin(2 * np.pi * f * t + 1.6 * np.exp(-t * 5) * np.sin(2 * np.pi * f * t * 2)) * np.exp(-t * 2.2)


def pluck(f, n, rng, cut=3500):
    t = tt(n); x = saw(f, n, rng.random()) * 0.6 + np.sin(2 * np.pi * f * t) * 0.4
    return sos_f(x, 'low', min(cut, f * 8 + 800)) * np.exp(-t * 7)


# ---------- composition ----------
def curve(kind, B):
    p = np.linspace(0, 1, B)
    if kind == 'build': I = 0.15 + 0.85 * p ** 0.8
    elif kind == 'wave': I = 0.22 + 0.75 * np.abs(np.sin(1.5 * np.pi * p))
    else: I = np.minimum(0.8, 0.25 + 1.5 * p)
    I[0] = min(I[0], 0.3)
    return np.clip(I, 0, 1)


def pick(clip, i):
    m = clip.get('music', {})
    h = int(hashlib.md5(clip['name'].encode()).hexdigest(), 16)
    style = m.get('style') or ORDER[(h + i) % len(ORDER)]
    return style, m.get('key'), m.get('seed', h % 10000)


def make_track(dur, style, key, seed):
    S = STYLES[style]
    rng = np.random.default_rng(seed)
    B = max(4, round(dur * S['bpm'] / 240))
    bar_s = dur / B                           # exact fit: bpm adjusts a few %
    bpm = 240 / bar_s
    N = int(dur * SR) + SR // 2
    bar_n = bar_s * SR
    step_n = bar_n / 16
    root = NOTES[key or S['keys'][seed % 3]] + 48
    scale = [root + x for x in (MINOR if S['mode'] == 'minor' else MAJOR)]
    scale += [x + 12 for x in scale] + [x + 24 for x in scale[:7]]
    prog = S['progs'][seed % len(S['progs'])]
    I = curve(S['curve'], B)

    dry, pads, send = Bus(N), Bus(N), Bus(N)    # pads bus gets sidechain ducking
    kicks = []
    one = dict(kick=kick(rng), snare=snare(rng), clap=clap(rng), hat=hat(rng), ohat=hat(rng, True),
               tom=tom(rng), tomlo=tom(rng, 70), boom=boom(rng), crash=crash(rng))

    def chord(deg):
        c = [scale[deg], scale[deg + 2], scale[deg + 4]]
        if S.get('ext7'): c.append(scale[deg + 6])
        if S.get('dissonance'): c.append(scale[deg + 1] - 12 + 12)
        return c

    # motif for the lead (2 bars, 8th notes), regenerated per piece
    motif = []
    d = 7
    for _ in range(16):
        d = int(np.clip(d + rng.choice([-2, -1, 0, 1, 2]), 3, 12))
        motif.append(d if rng.random() > 0.35 else None)

    for b in range(B):
        t0 = int(b * bar_n); inten = I[b]; last = b == B - 1
        deg = prog[b % len(prog)]
        notes = chord(deg)
        # --- pad
        n = int(bar_n * 1.15)
        pg = 0.5 + 0.5 * inten
        for side, pan in ((0, 0.25), (1, 0.75)):
            x = sum(supersaw(hz(m), n, rng, 0.009) for m in notes) / len(notes)
            x = sos_f(x, 'low', S['pad_cut'] * (0.5 + inten))
            x *= adsr(n, min(S['pad_att'], bar_s * 0.6), 0.1, 0.9, bar_s * 0.3)
            pads.add(t0, x, pan, 0.22 * pg)
            send.add(t0, x, pan, 0.12)
        # --- bass
        if inten >= 0.3 or last:
            f = hz(scale[deg] - 24)
            kind = S['bass']
            if last: kind = 'whole'
            if kind == 'whole':
                n = int(bar_n); x = (np.sin(2 * np.pi * f * tt(n)) + 0.3 * saw(f, n)) * adsr(n, 0.02, 0.1, 0.85, 0.2)
                pads.add(t0, sos_f(x, 'low', 400), 0.5, 0.45)
            else:
                pat = {'offbeat': [2, 6, 10, 14], 'eighths': list(range(0, 16, 2)), 'pulse': list(range(0, 16, 4)),
                       'syncop': [0, 3, 6, 10, 12]}[kind]
                for s in pat:
                    n = int(step_n * (3 if kind in ('syncop', 'pulse') else 1.7))
                    x = (np.sin(2 * np.pi * f * tt(n)) + 0.35 * saw(f, n)) * adsr(n, 0.005, 0.05, 0.6, 0.06)
                    pads.add(t0 + int(s * step_n), sos_f(x, 'low', 500), 0.5, 0.42)
        # --- drums
        dm = S['drums']
        if last:
            dry.add(t0, one['boom'], 0.5, 0.8); dry.add(t0, one['kick'], 0.5, 0.7); dry.add(t0, one['crash'], 0.5, 0.5)
            kicks.append(t0)
        elif inten >= 0.45 or (dm == 'sparse' and inten >= 0.3):
            P = {'four': dict(k=[0, 4, 8, 12], sn=[4, 12], h=[2, 6, 10, 14], hc=[0, 2, 4, 6, 8, 10, 12, 14]),
                 'pop': dict(k=[0, 8, 10], sn=[4, 12], h=list(range(0, 16, 2)), hc=[]),
                 'halftime': dict(k=[0, 10], sn=[8], h=list(range(0, 16, 2)), hc=[]),
                 'lofi': dict(k=[0, 7, 10], sn=[4, 12], h=list(range(0, 16, 2)), hc=[]),
                 'cinematic': dict(k=[0], sn=[], h=[], hc=[], tm=[0, 6, 10], tl=[8]),
                 'sparse': dict(k=[0], sn=[], h=[4, 12], hc=[])}[dm]
            swing = 0.12 if dm == 'lofi' else 0.0
            for s in P['k']:
                p = t0 + int(s * step_n); dry.add(p, one['kick'], 0.5, 0.75 if dm != 'cinematic' else 0.9); kicks.append(p)
            if inten >= 0.55:
                for s in P['sn']:
                    p = t0 + int(s * step_n)
                    dry.add(p, one['clap' if dm in ('four', 'pop') else 'snare'], 0.5, 0.5); send.add(p, one['snare'], 0.5, 0.2)
                for s in P['h']:
                    dry.add(t0 + int((s + (swing if s % 2 else 0)) * step_n), one['hat'], 0.65, 0.5 * inten)
                for s in P['hc']:
                    if s % 4: dry.add(t0 + int(s * step_n), one['hat'], 0.4, 0.3 * inten)
            for s in P.get('tm', []):
                dry.add(t0 + int(s * step_n), one['tom'], 0.4, 0.6 * inten)
            for s in P.get('tl', []):
                dry.add(t0 + int(s * step_n), one['tomlo'], 0.6, 0.7 * inten)
        # --- arp / pluck pattern
        if S['arp'] and inten >= 0.35 and not last:
            rate = {'pluck16': 1, 'pluck8': 2, 'tick': 4}[S['arp']]
            idx = [0, 1, 2, 3, 2, 1, 2, 3] if b % 2 == 0 else [2, 3, 1, 3, 0, 2, 1, 3]
            ext = notes + [notes[0] + 12, notes[1] + 12]
            for j, s in enumerate(range(0, 16, rate)):
                m = ext[idx[j % 8] % len(ext)] + (12 if S['arp'] != 'tick' else 24)
                if S['arp'] == 'tick' and rng.random() > 0.5: continue
                n = int(step_n * rate * 1.6)
                x = pluck(hz(m), n, rng) * (0.14 + 0.1 * inten)
                pads.add(t0 + int(s * step_n), x, 0.3 + 0.4 * ((j * 7) % 3) / 2, 1.0)
                send.add(t0 + int(s * step_n), x, 0.5, 0.35)
        # --- electric piano chords (chill)
        if S.get('keys_ep') and not last:
            for s in [0, 6, 10] if inten > 0.4 else [0]:
                for m in notes:
                    x = ep_note(hz(m + 12), int(bar_n * 0.55)) * 0.07
                    dry.add(t0 + int(s * step_n), x, 0.5 + rng.uniform(-.2, .2), 1.0)
                    send.add(t0 + int(s * step_n), x, 0.5, 0.3)
        # --- lead melody (2-bar motif, last repeat varies)
        if S['lead'] and inten >= 0.6 and not last:
            half = b % 2
            for j in range(8):
                dd = motif[half * 8 + j]
                if dd is None: continue
                if b % 4 == 3 and j > 4: dd = max(3, dd - 2)
                m = scale[dd] + (12 if S['lead'] != 'ep' else 0)
                n = int(step_n * 2 * 1.8)
                if S['lead'] == 'saw':
                    x = sos_f(supersaw(hz(m), n, rng, 0.006, 3), 'low', 2600) * adsr(n, 0.02, 0.1, 0.7, 0.25) * 0.16
                elif S['lead'] == 'ep':
                    x = ep_note(hz(m), n) * 0.11
                else:
                    x = pluck(hz(m), n, rng, 5000) * 0.2
                dry.add(t0 + int(j * 2 * step_n), x, 0.55, 1.0); send.add(t0 + int(j * 2 * step_n), x, 0.5, 0.4)
        # --- stabs (trailer)
        if S['stabs'] and inten >= 0.8 and not last:
            for s in (0, 3, 6):
                n = int(step_n * 2)
                x = sos_f(sum(supersaw(hz(m), n, rng, 0.012) for m in notes[:3]) / 3, 'low', 2200) * adsr(n, 0.005, 0.1, 0.5, 0.1) * 0.2
                dry.add(t0 + int(s * step_n), x, 0.5, 1.0)
        # --- riser into the strongest part + impact
        if S['riser'] and b == int(np.argmax(I[:-1])) and b >= 2:
            L = 2 * bar_s; st = max(0, t0 - int(L * SR))
            send.add(st, riser(rng, L), 0.5, 0.6)
            dry.add(t0, one['boom'], 0.5, 0.5); dry.add(t0, one['crash'], 0.5, 0.35)

    # sidechain ducking of pads/bass/arps on kicks
    if S['sc'] > 0 and kicks:
        sc = np.ones(N); m = int(0.22 * SR); duck = 1 - S['sc'] * np.exp(-tt(m) / 0.09)
        for p in kicks:
            e = min(N, p + m)
            if e > p: sc[p:e] = np.minimum(sc[p:e], duck[:e - p])
        pads.a *= sc
    # reverb on the send bus
    irn = int(1.8 * SR); ir = rng.standard_normal((2, irn)) * np.exp(-tt(irn) * 3.2)
    wet = np.stack([signal.fftconvolve(send.a[c], ir[c])[:N] for c in range(2)])
    wet = sos_f(wet, 'low', 7000) * (S['verb'] * 0.08)
    mix = pads.a + dry.a + wet
    if S.get('vinyl'):
        crackle = (rng.random(N) > 0.9995) * rng.standard_normal(N) * 0.05
        mix += sos_f(rng.standard_normal(N), 'band', [1000, 6000]) * 0.003 + crackle
    mix = sos_f(mix, 'high', 30)
    mix = mix[:, :int(dur * SR)]
    mix *= 0.13 / (np.sqrt(np.mean(mix ** 2)) + 1e-9)       # about -17 dB RMS (like the narration)
    mix = np.tanh(mix * 1.4) / np.tanh(1.4) * 0.97            # soft limiter
    return mix, bpm, B


def write_wav(path, mix):
    pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


if __name__ == '__main__':
    clips = json.load(open(sys.argv[1]))['clips']; out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    prev = None
    for i, c in enumerate(clips, 1):
        style, key, seed = pick(c, i)
        if style == prev and not c.get('music', {}).get('style'):
            style = ORDER[(ORDER.index(style) + 1) % len(ORDER)]
        prev = style
        dur = c['end'] - c['start']
        mix, bpm, B = make_track(dur, style, key, seed)
        path = os.path.join(out, f'{i:02d}_{c["name"]}.wav')
        write_wav(path, mix)
        print(f'{i:02d} {c["name"]:28s} {dur:5.1f}s style={style:8s} bpm={bpm:5.1f} bars={B}', flush=True)
