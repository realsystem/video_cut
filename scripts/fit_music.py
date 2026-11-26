"""Fit REAL music tracks to the shorts so the beat lands on the video's scene cuts.

usage: python fit_music.py CLIPS.json TRACKS_DIR SHORTS_DIR OUT_DIR [--max-stretch 0.06] [--gain-lufs -16]

TRACKS_DIR holds audio files (mp3/wav/m4a/ogg/flac) you picked and licensed.
  - A file named like "03_anything.mp3" is used for clip 03 only.
  - Unnumbered files are shared round-robin; one track can serve several clips (different sections).
For each clip the tool:
  1. finds the track's tempo + beat grid (constant tempo assumed; works for most rock/electronic),
  2. picks a loud, beat-aligned section that doesn't repeat sections already used,
  3. speeds/slows it by <= max-stretch (pitch preserved) so beats fall on the clip's inner scene cuts,
  4. loudness-normalises, fades out, and muxes it onto SHORTS_DIR/NN_*.mp4 (video stream copied).
Prints a report with the average beat-to-cut error per clip. Needs ffmpeg + numpy + scipy.
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
from scipy import signal

SR, HOP, NFFT = 22050, 512, 1024
FR = SR / HOP  # onset frames per second


def load(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def onset_env(x):
    _, _, Z = signal.stft(x, SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
    S = np.log1p(10 * np.abs(Z))
    flux = np.maximum(0, np.diff(S, axis=1)).sum(0)
    flux = np.concatenate([[0], flux])
    flux -= np.convolve(flux, np.ones(43) / 43, mode='same')      # remove slow trend
    return np.maximum(flux, 0)


def beat_grid(env):
    """Return (period_s, phase_s) of the best constant beat grid + a confidence number."""
    n = len(env); t = np.arange(n) / FR
    ac = np.correlate(env - env.mean(), env - env.mean(), 'full')[n - 1:]
    lags = np.arange(len(ac)) / FR
    lo, hi = 60 / 190, 60 / 75
    m = (lags >= lo) & (lags <= hi)
    # candidate periods: the strongest autocorrelation peaks (+ musical ratios of them)
    idx = np.where(m)[0]
    pk, _ = signal.find_peaks(ac[idx])
    pk = idx[pk][np.argsort(ac[idx][signal.find_peaks(ac[idx])[0]])[::-1][:6]]
    cands = set()
    for p in pk:
        for r in (1, 2, 0.5):
            if lo <= lags[p] * r <= hi: cands.add(lags[p] * r)
    best = (-1, None, 0.0, 0.0)
    for T0 in cands:
        for T in np.linspace(T0 * 0.96, T0 * 1.04, 161):        # fine search over the whole track
            k = np.floor(t / T).astype(int); ph = (t - k * T) / T
            bins = np.minimum((ph * 48).astype(int), 47)
            hist = np.bincount(bins, weights=env, minlength=48)
            hist = hist + np.roll(hist, 1) * 0.5 + np.roll(hist, -1) * 0.5
            frac = hist.max() / (hist.sum() + 1e-9)             # share of onset energy sitting on the grid
            prior = np.exp(-0.5 * (np.log2((60 / T) / 125) / 1.2) ** 2)   # mild preference for 90-170 bpm
            if frac * prior > best[0]:
                best = (frac * prior, T, (np.argmax(hist) + 0.5) / 48 * T, frac)
    return best[1], best[2], best[3]


def analyse(path):
    x = load(path); env = onset_env(x)
    T, ph, conf = beat_grid(env)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(SR // 4) / (SR // 4), mode='same'))
    return dict(path=path, dur=len(x) / SR, T=T, ph=ph, conf=conf, rms=rms[::SR // 20], rms_rate=20)


def plan_clip(tr, cuts, dur, max_stretch, used):
    """cuts: scene-cut times inside the clip (seconds from clip start). Returns best plan dict."""
    best = None
    for s in np.linspace(1 - max_stretch, 1 + max_stretch, 25):    # s>1 = faster
        T = tr['T'] / s
        for phi in np.arange(0, T, T / 32):                         # output beat positions: phi + k*T
            if len(cuts):
                d = [abs(((c - phi + T / 2) % T) - T / 2) for c in cuts]
                cost = float(np.mean(np.minimum(d, 0.12))) / 0.12
            else:
                cost = 0.0
            cost += 4 * abs(s - 1)
            # candidate source beats whose window is loud and new
            beats = tr['ph'] + np.arange(0, int(tr['dur'] / tr['T']) + 1) * tr['T']
            for bj in beats[8::2]:
                ts = bj - phi * s
                te = ts + dur * s
                if ts < 0 or te > tr['dur'] - 0.2: continue
                i0, i1 = int(ts * tr['rms_rate']), int(te * tr['rms_rate'])
                seg = tr['rms'][i0:i1]
                if len(seg) < 4: continue
                en = float(seg.mean()) + 0.5 * float(seg[:60].mean())
                reuse = sum(max(0, min(te, b) - max(ts, a)) for a, b in used.get(tr['path'], [])) / (te - ts)
                score = cost * 0.6 - en * 6 + reuse * 3
                if best is None or score < best['score']:
                    best = dict(score=score, s=s, ts=ts, te=te, phi=phi, T=T, cost=cost)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clips'); ap.add_argument('tracks'); ap.add_argument('shorts'); ap.add_argument('out')
    ap.add_argument('--max-stretch', type=float, default=0.06); ap.add_argument('--gain-lufs', type=float, default=-16)
    a = ap.parse_args()
    cfg = json.load(open(a.clips)); clips, cuts_all = cfg['clips'], cfg['scene_cuts']
    files = sorted(sum((glob.glob(os.path.join(a.tracks, e)) for e in ('*.mp3', '*.wav', '*.m4a', '*.ogg', '*.flac')), []))
    if not files: sys.exit('no audio files found in ' + a.tracks)
    ana = {}
    for f in files:
        ana[f] = analyse(f)
        print(f'track {os.path.basename(f)}: {60/ana[f]["T"]:.1f} bpm, {ana[f]["dur"]:.0f}s, beat confidence {ana[f]["conf"]:.2f}')
    numbered = {int(os.path.basename(f)[:2]): f for f in files if os.path.basename(f)[:2].isdigit()}
    shared = [f for f in files if f not in numbered.values()] or files
    used, rr = {}, 0
    os.makedirs(a.out, exist_ok=True)
    for i, c in enumerate(clips, 1):
        dur = c['end'] - c['start']
        if i in numbered: f = numbered[i]
        else: f = shared[rr % len(shared)]; rr += 1
        cuts = [x - c['start'] for x in cuts_all if 0.3 < x - c['start'] < dur - 0.3]
        p = plan_clip(ana[f], cuts, dur, a.max_stretch, used)
        if p is None: print(f'{i:02d} no usable section in {os.path.basename(f)} (too short?)'); continue
        used.setdefault(f, []).append((p['ts'], p['te']))
        v = glob.glob(os.path.join(a.shorts, f'{i:02d}_*.mp4'))[0]
        af = (f'atempo={p["s"]:.5f},loudnorm=I={a.gain_lufs}:TP=-1.5:LRA=11,'
              f'afade=t=in:d=0.3,afade=t=out:st={dur-1.5:.3f}:d=1.5')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', v, '-ss', f'{p["ts"]:.3f}', '-t', f'{p["te"]-p["ts"]:.3f}', '-i', f,
                        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-af', af, '-c:a', 'aac', '-b:a', '192k',
                        '-t', f'{dur:.3f}', '-movflags', '+faststart', os.path.join(a.out, os.path.basename(v))], check=True)
        err = [min(abs(((x - p['phi'] + p['T'] / 2) % p['T']) - p['T'] / 2) for _ in [0]) * 1000 for x in cuts]
        print(f'{i:02d} {os.path.basename(f)[:28]:28s} from {p["ts"]:6.1f}s speed {100*(p["s"]-1):+4.1f}% '
              f'cut->beat error {np.mean(err) if err else 0:4.0f} ms over {len(cuts)} cuts')


if __name__ == '__main__':
    main()
