"""Environment classifier — OWNER: Person 2.

Input:  ExtractedInputs (audio_path, frames_dir, fps, duration)
Output: list[EnvSegment] covering [0, duration]  ->  env_segments.json

    segmentation.detect_spans   video -> [(start, end)]       (PySceneDetect)
    audio.audio_env_logits      audio  x spans -> (S, E)      (CLAP prompts)
    visual.visual_env_logits    frames x spans -> (S, E)      (CLIP/SigLIP prompts)
    fusion.fuse + to_segments   -> list[EnvSegment] with `general` fallback
"""
