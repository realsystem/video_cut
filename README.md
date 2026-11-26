# video_cut

Turn a long landscape video into vertical (9:16) shorts, entirely on a local machine:
`yt-dlp` download, scene-aligned cuts, subject-tracked crop, generated music, offline QA.
Built on Windows 10 (PowerShell) with ffmpeg, Python 3.15.

## Pipeline

```
1. download   python -m yt_dlp -f "bv*+ba/b" --merge-output-format mkv -o "source_best.%(ext)s" <URL>
2. scene cuts ffmpeg -i source_best.mkv -vf "select='gt(scene,0.25)',showinfo" -an -f null -
              (pts_time values -> clips.json "scene_cuts"; pick clips between cuts)
3. music      python scripts/fit_music.py clips.json TRACKS_DIR SHORTS_DIR OUT_DIR  -> real tracks you chose,
              beat-fitted to each clip's scene cuts (recommended; see Music below)
4. track      python scripts/track.py source_best.mkv clips.json CMD_DIR
5. render     python scripts/render.py source_best.mkv clips.json CMD_DIR MUSIC_DIR OUT_DIR
6. check      python scripts/check.py OUT_DIR
```

Requirements: `ffmpeg`/`ffprobe` on PATH, `pip install yt-dlp[default] pip-system-certs numpy scipy ultralytics`.

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
- Music is synthesized (`musicgen.py`, numpy/scipy), so it's original and license-free. Every clip
  gets its own track: style (`epic|driving|tense|uplift|chill|trailer`), key and seed come from the
  clip's `"music"` entry in `clips.json` (auto-picked if absent, adjacent clips never share a style).
  Tempo is nudged so whole bars fit the clip exactly; each track builds through an intensity arc
  and ends on a boom/crash at the clip's end. It still sounds synthetic: swap in real
  royalty-free files if wanted. To change only the music of rendered shorts (no video re-encode):
  `python scripts/remux_audio.py clips.json SHORTS_DIR MUSIC_DIR OUT_DIR [GAIN_DB]`.
- Music was judged by measurement only (levels, loudness arc, spectrum), not by ear: listen to it.

## Music: what worked and what didn't (listening verdict from the channel owner)

- `musicgen.py` (numpy synth, styles epic/driving/tense/uplift/chill/trailer + industrial/metal/rock/phonk/
  wobble): rejected: "ancient synthesizer", no real bass, all tracks sound alike, unrelated to the picture.
  Fine as a placeholder generator, not for publishing.
- `stable_audio.py` (Stable Audio Open 1.0 on CPU, ~12 min/clip): rejected: generic/ambient, unrelated
  to hard off-road content; also fades out near the end of each generation and has a restrictive license.
- ACE-Step 1.5 (MIT, 4B/2B on CPU) was never tested: the Claude Code memory guard killed it while
  loading/sampling on a 32 GB machine (CPU runs fp32). Needs a real GPU. `acestep_gen.py`/`tsrun.py` are not
  in the repo for that reason.
- Genre research (blogs, weak evidence): rock/hard rock/metal for rock crawling; phonk trending for car edits.
- **Recommended:** pick real royalty-free tracks by ear, then `fit_music.py`: detects tempo + beat grid
  (verified exact on synthetic tracks with known bpm), chooses a loud section, stretches speed by <= 6%
  (pitch preserved) so beats land on the clip's inner scene cuts, loudness-normalises, fades out and muxes
  onto the existing video without re-encoding. Assumes constant tempo; check the printed cut->beat error.
- QA: `check.py` is numeric only (scene cuts at clip edges, vehicle presence, edge-touch,
  audio levels) so no frames need to be looked at manually.

## Workflow for a new video

1. Publish the main video, download it in best quality (step 1).
2. Run scene detection (step 2) and pick clips between cuts; write them to a new `clips.json`
   with a `music.style` that fits each moment (tense for trouble, epic for scenery, ...).
3. Steps 3-6; listen to the clips, adjust styles/seeds, re-run only `musicgen.py` + `remux_audio.py`.

## Not in the repo

Video files, YOLO weights (`yolov8n.pt`, auto-downloaded) and generated `cmd*.txt` / `.wav` files.
