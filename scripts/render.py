"""Pass 2: render 1080x1920 shorts from the 4K source with the tracked crop + music.

usage: python render.py SOURCE.mkv CLIPS.json CMD_DIR MUSIC_DIR OUT_DIR
CMD_DIR  = output of track.py; MUSIC_DIR = output of mkmusic.py (trackA/B/C.wav)
"""
import json, os, subprocess, sys

src, clips_json, cmd_dir, music_dir, out = sys.argv[1:6]
SRC_H = 2160
CW = int(SRC_H * 9 / 16) // 2 * 2
os.makedirs(out, exist_ok=True)
for i, c in enumerate(json.load(open(clips_json))['clips'], 1):
    d = round(c['end'] - c['start'], 3)
    music = os.path.join(music_dir, 'track' + 'ABC'[(i - 1) % 3] + '.wav')
    cmd_file = f'cmd{i:02d}.txt'  # relative: sendcmd paths break on Windows drive letters
    fc = (f'[0:v]sendcmd=f={cmd_file},crop={CW}:{SRC_H}:0:0,scale=1080:1920:flags=lanczos,'
          f'setsar=1,fps=60000/1001[v];[1:a]afade=t=in:d=0.5,afade=t=out:st={d-1.2}:d=1.2[a]')
    dst = os.path.abspath(os.path.join(out, f'{i:02d}_{c["name"]}.mp4'))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(c['start']), '-t', str(d),
                    '-i', os.path.abspath(src), '-i', os.path.abspath(music), '-t', str(d),
                    '-filter_complex', fc, '-map', '[v]', '-map', '[a]',
                    '-c:v', 'libx264', '-crf', '17', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', dst],
                   cwd=cmd_dir, check=True)
    print('rendered', dst)
