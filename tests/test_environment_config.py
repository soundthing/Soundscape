"""The shared environment IDs must remain usable by every pipeline stage."""


def test_expanded_environment_config(cfg):
    added = {"lecture_hall", "restaurant", "cafe", "public_transportation",
             "factory", "construction", "classroom"}
    assert added <= set(cfg.env_ids)
    for env in cfg.environments:
        if env.id in added:
            assert env.audio_prompts and env.visual_prompts
            assert env.id in cfg.priors.environments
