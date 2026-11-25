"""Swap the audio of already-rendered shorts for per-clip music without re-encoding the video.

usage: python remux_audio.py CLIPS.json SHORTS_DIR MUSIC_DIR OUT_DIR [GAIN_DB]
Matches files by index: SHORTS_DIR/NN_*.mp4 + MUSIC_DIR/NN_*.wav (from musicgen.py).
"""
import glob, json, os, subprocess, sys

clips_json, shorts, music, out = sys.argv[1:5]
gain = sys.argv[5] if len(sys.argv) > 5 else '-2'
os.makedirs(out, exist_ok=True)
for i, c in enumerate(json.load(open(clips_json))['clips'], 1):
    v = glob.glob(os.path.join(shorts, f'{i:02d}_*.mp4'))[0]
    a = glob.glob(os.path.join(music, f'{i:02d}_*.wav'))[0]
    d = round(c['end'] - c['start'], 3)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', v, '-i', a, '-map', '0:v', '-map', '1:a',
                    '-c:v', 'copy', '-af', f'volume={gain}dB,afade=t=in:d=0.5,afade=t=out:st={d-1.2}:d=1.2',
                    '-c:a', 'aac', '-b:a', '192k', '-t', str(d), '-movflags', '+faststart',
                    os.path.join(out, os.path.basename(v))], check=True)
    print('remuxed', os.path.basename(v))
