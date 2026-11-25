"""Offline QA for rendered shorts (no images are viewed by anyone).

usage: python check.py SHORTS_DIR
Reports per clip: duration/format, scene cuts in the first/last 0.4 s (a cut there means a
stray frame from the neighbouring scene), % frames with no vehicle detected, % frames where
the main vehicle touches the left/right edge (truncated by the crop), audio mean/max dB.
"""
import glob, os, re, subprocess, sys
import numpy as np
from ultralytics import YOLO

D = sys.argv[1]
W, H = 540, 960
model = YOLO('yolov8n.pt')


def run(*a, **k):
    return subprocess.run(list(a), capture_output=True, **k)


for f in sorted(glob.glob(os.path.join(D, '*.mp4'))):
    dur = float(run('ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', f, text=True).stdout)
    fmt = run('ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
              'stream=width,height,r_frame_rate', '-of', 'csv=p=0', f, text=True).stdout.strip()
    err = run('ffmpeg', '-i', f, '-vf', "select='gt(scene,0.25)',showinfo", '-an', '-f', 'null', '-', text=True).stderr
    cuts = [round(float(m), 2) for m in re.findall(r'pts_time:([\d.]+)', err)]
    edge_cuts = [c for c in cuts if c < 0.4 or c > dur - 0.4]
    au = run('ffmpeg', '-i', f, '-vn', '-af', 'volumedetect', '-f', 'null', '-', text=True).stderr
    mean, mx = (re.search(p + r': ([-\d.]+)', au).group(1) for p in ('mean_volume', 'max_volume'))
    raw = run('ffmpeg', '-v', 'error', '-i', f, '-vf', f'fps=5,scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-').stdout
    n = len(raw) // (W * H * 3)
    fr = np.frombuffer(raw[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)
    none = edge = 0
    for im in fr:
        r = model(im, verbose=False, classes=[2, 5, 7], conf=0.25)[0]
        if len(r.boxes) == 0:
            none += 1
            continue
        b = max(r.boxes, key=lambda b: b.xywh[0][2] * b.xywh[0][3])
        x1, _, x2, _ = b.xyxy[0].tolist()
        edge += x1 <= 3 or x2 >= W - 3
    print(f'{os.path.basename(f)} | {dur:.1f}s | {fmt} | edge cuts: {edge_cuts or "none"} | '
          f'no vehicle {100*none/n:.0f}% | touches edge {100*edge/n:.0f}% | audio {mean}/{mx} dB')
