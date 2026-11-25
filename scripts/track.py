"""Pass 1: find the main vehicle in each clip and write per-frame crop-x commands
(ffmpeg sendcmd files) so a 9:16 window follows it.

usage: python track.py SOURCE.mkv CLIPS.json OUT_DIR
Requires: ffmpeg on PATH, numpy, ultralytics.
"""
import json, subprocess, sys
import numpy as np
from ultralytics import YOLO

src, clips_json, out = sys.argv[1:4]
cfg = json.load(open(clips_json))
CUTS, CLIPS = cfg['scene_cuts'], cfg['clips']
W, H, FPS_DET = 1280, 720, 10
SRC_W, SRC_H = 3840, 2160            # adjust if the source is not 4K
CW = int(SRC_H * 9 / 16) // 2 * 2    # crop width for 9:16
model = YOLO('yolov8n.pt')            # downloads weights on first run

for ci, c in enumerate(CLIPS, 1):
    a, d = c['start'], c['end'] - c['start']
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(a), '-t', str(d), '-i', src,
           '-vf', f'fps={FPS_DET},scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (W * H * 3)
    frames = np.frombuffer(raw[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)
    centers = []
    for fr in frames:  # largest car/bus/truck box = subject
        r = model(fr, verbose=False, classes=[2, 5, 7], conf=0.25)[0]
        best = None
        for bx in r.boxes:
            x1, _, x2, _ = bx.xyxy[0].tolist()
            area = (x2 - x1) * bx.xywh[0][3].item()
            if best is None or area > best[0]:
                best = (area, (x1 + x2) / 2)
        centers.append(best[1] if best else np.nan)
    cen = np.array(centers, float)
    idx = np.arange(n)
    ok = ~np.isnan(cen)
    cen = np.interp(idx, idx[ok], cen[ok]) if ok.any() else np.full(n, W / 2)
    # smooth inside each scene only, so the crop jumps (not pans) across cuts
    scene = np.searchsorted(CUTS, a + idx / FPS_DET)
    sm = cen.copy()
    for s in np.unique(scene):
        m = scene == s
        v, k = cen[m], min(m.sum(), 15) | 1
        if len(v) >= 3:
            v = np.convolve(np.pad(v, k // 2, mode='edge'), np.ones(k) / k, mode='valid')[:len(v)]
        sm[m] = v
    fps = 60000 / 1001
    tf = np.arange(int(d * fps)) / fps
    x = np.clip(np.interp(tf, idx / FPS_DET, sm) * (SRC_W / W) - CW / 2, 0, SRC_W - CW).round().astype(int)
    x -= x % 2
    with open(f'{out}/cmd{ci:02d}.txt', 'w') as f:
        for t, xi in zip(tf, x):
            f.write(f'{t:.5f} crop x {xi};\n')
    print(ci, c['name'], 'frames', n, 'x range', int(x.min()), int(x.max()), flush=True)
