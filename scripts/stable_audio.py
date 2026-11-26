"""Generate one music track per clip with Stable Audio Open (local, CPU ok but slow).

usage: python stable_audio.py CLIPS.json OUT_DIR [--only N ...] [--steps 100] [--seed 0]
Writes OUT_DIR/NN_<clip-name>.wav (44.1 kHz stereo 16-bit), exactly the clip's duration.
Prompt = clip["music"]["prompt"] if present, else a template chosen by clip["music"]["style"].
Needs: pip install torch diffusers transformers accelerate soundfile, and a Hugging Face token
that has access to the gated repo stabilityai/stable-audio-open-1.0 (accept its license first).
Model limit is ~47 s of audio. Check the model license before publishing generated audio.
"""
import argparse, json, os, time
import numpy as np, soundfile as sf, torch
from diffusers import StableAudioPipeline

STYLE_PROMPTS = {
    'epic': 'Cinematic orchestral adventure score, building strings, deep taiko drums, wide open mountain landscape, hopeful and epic, instrumental',
    'tense': 'Tense suspense cinematic underscore, pulsing low synth, ticking percussion, rising tension, dark and uneasy, instrumental',
    'driving': 'Driving energetic rock electronic instrumental, punchy drums, gritty bass, off-road adventure, building intensity',
    'uplift': 'Uplifting bright instrumental, warm guitars and synth plucks, steady beat, optimistic road trip feeling',
    'chill': 'Chill laid-back instrumental groove, warm electric piano, soft drums, relaxed and thoughtful, lo-fi texture',
    'trailer': 'Powerful cinematic trailer music, heavy drums, low brass stabs, rising riser, determined and relentless, instrumental',
}

ap = argparse.ArgumentParser()
ap.add_argument('clips'); ap.add_argument('out')
ap.add_argument('--only', type=int, nargs='*'); ap.add_argument('--steps', type=int, default=100)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
clips = json.load(open(a.clips))['clips']
pipe = StableAudioPipeline.from_pretrained('stabilityai/stable-audio-open-1.0', torch_dtype=torch.float32).to('cpu')
sr = pipe.vae.sampling_rate
for i, c in enumerate(clips, 1):
    if a.only and i not in a.only: continue
    m = c.get('music', {})
    dur = c['end'] - c['start']
    prompt = m.get('prompt') or STYLE_PROMPTS[m.get('style', 'epic')]
    t0 = time.time()
    g = torch.Generator('cpu').manual_seed(a.seed + i)
    audio = pipe(prompt=prompt, negative_prompt='Low quality, vocals, speech, noise.',
                 num_inference_steps=a.steps, audio_end_in_s=min(dur + 4.0, 47.0),  # extra tail: the model fades out at its end, we trim it off
                 num_waveforms_per_prompt=1, generator=g).audios[0]
    x = audio.T.float().cpu().numpy()[:int(dur * sr)]
    pk = np.abs(x).max()
    if pk > 0: x = x / pk * 0.89
    path = os.path.join(a.out, f'{i:02d}_{c["name"]}.wav')
    sf.write(path, x, sr, subtype='PCM_16')
    print(f'{i:02d} {c["name"]} {dur:.1f}s generated in {time.time()-t0:.0f}s -> {path}', flush=True)
