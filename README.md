# video_cut

Turn a long landscape video into vertical (9:16) shorts, entirely on a local machine:
`yt-dlp` download, scene-aligned cuts, subject-tracked crop, generated music, offline QA.
Built on Windows 10 (PowerShell) with ffmpeg, Python 3.15.

## Pipeline

```
1. download   python -m yt_dlp -f "bv*+ba/b" --merge-output-format mkv -o "source_best.%(ext)s" <URL>
2. scene cuts ffmpeg -i source_best.mkv -vf "select='gt(scene,0.25)',showinfo" -an -f null -
              (pts_time values -> clips.json "scene_cuts"; pick clips between cuts)
3. music      python scripts/mkmusic.py MUSIC_DIR          -> trackA/B/C.wav (original synth tracks)
4. track      python scripts/track.py source_best.mkv clips.json CMD_DIR
5. render     python scripts/render.py source_best.mkv clips.json CMD_DIR MUSIC_DIR OUT_DIR
6. check      python scripts/check.py OUT_DIR
```

Requirements: `ffmpeg`/`ffprobe` on PATH, `pip install yt-dlp[default] pip-system-certs numpy ultralytics`.

## Lessons / gotchas

- **Download the best quality explicitly.** Capping at `height<=1080` silently gave 1080p when
  the source had 2160p60 (VP9). Use `-F` to list formats and check with `ffprobe` afterwards.
- **SSL `CERTIFICATE_VERIFY_FAILED` on Windows Python:** `pip install pip-system-certs`
  (uses the Windows cert store). Don't disable verification.
- **Cut on scene changes, not on subtitle/sentence boundaries.** Sentence-aligned cuts ended
  mid-action. Use exact detector timestamps, and end each clip ~0.02 s *before* the next
  cut, otherwise one frame of the next scene leaks into the end (rounded times did this to
  8 of 10 clips; only measurable with the scene detector on the output, not by eye).
- **Center crop to 9:16 cuts off subjects.** The tracker (YOLOv8n, largest car/truck box,
  smoothed per scene, jump at cuts) drives a per-frame `crop x` through `sendcmd`.
  `sendcmd=f=` needs a relative path on Windows (drive-letter colon breaks parsing), so
  `render.py` runs ffmpeg with `cwd=CMD_DIR`.
- **Close-ups are wider than a 9:16 slice** (80-94% of frames in some clips touch the edge).
  Tracking can't fix that. Idea not yet done: fit the full frame over a blurred background
  for those shots.
- YOLOv8n misses low front-on truck angles (34% no-detection in one clip); the crop falls
  back to the last/interpolated position there.
- Music is synthesized (stdlib only), so it's original and license-free. It sounds synthetic;
  swap in real royalty-free files if wanted (mux with the same `afade` settings).
- QA: `check.py` is numeric only (scene cuts at clip edges, vehicle presence, edge-touch,
  audio levels) so no frames need to be looked at manually.

## Not in the repo

Video files, YOLO weights (`yolov8n.pt`, auto-downloaded) and generated `cmd*.txt` / `.wav` files.
