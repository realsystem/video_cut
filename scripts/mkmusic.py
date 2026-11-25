import math, random, struct, wave, sys
SR = 44100

def hz(m): return 440.0 * 2 ** ((m - 69) / 12)

def render(path, bpm, prog, minor, dur, seed):
    random.seed(seed)
    n = int(dur * SR)
    L = [0.0] * n
    R = [0.0] * n
    beat = 60.0 / bpm
    bar = beat * 4
    third = 3 if minor else 4
    for b in range(int(dur / bar) + 1):
        root = prog[b % len(prog)]
        t0 = b * bar
        s0 = int(t0 * SR)
        chord = [root, root + third, root + 7, root + 12]
        # pad: detuned soft saws
        plen = int(bar * SR)
        for k, note in enumerate(chord):
            f = hz(note + 12)
            for det in (-0.004, 0.004):
                ff = f * (1 + det)
                pan = 0.3 if det < 0 else 0.7
                for i in range(plen):
                    j = s0 + i
                    if j >= n: break
                    t = i / SR
                    env = min(1, t / 0.6) * min(1, (bar - t) / 0.5)
                    ph = (ff * t) % 1.0
                    v = (2 * ph - 1) * 0.5 + math.sin(6.2832 * ff * t) * 0.5
                    v *= 0.025 * env
                    L[j] += v * (1 - pan)
                    R[j] += v * pan
        # bass: sine on root, per beat pulse
        for q in range(4):
            bs = s0 + int(q * beat * SR)
            f = hz(root - 12)
            for i in range(int(beat * SR * 0.9)):
                j = bs + i
                if j >= n: break
                t = i / SR
                env = math.exp(-t * 3.0) * min(1, t / 0.01)
                v = math.sin(6.2832 * f * t) * 0.22 * env
                L[j] += v; R[j] += v
        # arpeggio pluck on 8ths
        seq = [chord[0], chord[2], chord[3], chord[2], chord[1], chord[2], chord[3], chord[1]]
        for q in range(8):
            bs = s0 + int(q * beat / 2 * SR)
            f = hz(seq[q] + 24)
            for i in range(int(beat * SR * 0.5)):
                j = bs + i
                if j >= n: break
                t = i / SR
                env = math.exp(-t * 9.0)
                v = (math.sin(6.2832 * f * t) + 0.3 * math.sin(12.566 * f * t)) * 0.05 * env
                pan = 0.35 + 0.3 * (q % 2)
                L[j] += v * (1 - pan) * 2; R[j] += v * pan * 2
        # kick on beats, snare-ish noise on 2 and 4, hats on off-8ths
        for q in range(4):
            bs = s0 + int(q * beat * SR)
            for i in range(int(0.25 * SR)):
                j = bs + i
                if j >= n: break
                t = i / SR
                f = 45 + 90 * math.exp(-t * 30)
                v = math.sin(6.2832 * f * t) * math.exp(-t * 14) * 0.45
                if q % 2 == 1:
                    v += (random.random() * 2 - 1) * math.exp(-t * 18) * 0.10
                L[j] += v; R[j] += v
        for q in range(4):
            bs = s0 + int((q + 0.5) * beat * SR)
            for i in range(int(0.05 * SR)):
                j = bs + i
                if j >= n: break
                t = i / SR
                v = (random.random() * 2 - 1) * math.exp(-t * 90) * 0.05
                L[j] += v; R[j] += v
    peak = max(max(map(abs, L)), max(map(abs, R)), 1e-9)
    g = 0.89 / peak
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        buf = bytearray()
        for i in range(n):
            buf += struct.pack('<hh', int(L[i] * g * 32767), int(R[i] * g * 32767))
        w.writeframes(bytes(buf))

out = sys.argv[1]
# A minor, 100bpm / D minor 112bpm / G major 108bpm  (MIDI roots)
render(out + '/trackA.wav', 100, [57, 53, 48, 55], True, 48, 1)
render(out + '/trackB.wav', 112, [50, 46, 53, 48], True, 48, 2)
render(out + '/trackC.wav', 108, [55, 52, 48, 50], False, 48, 3)
print('ok')
