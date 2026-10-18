"""Registry of harness components. Modules are imported lazily by `get`."""

from __future__ import annotations

import importlib

from soundthing.harness.components.base import Clip, Component, Evaluation, is_config_key

# name -> "module:attribute" of a Component instance
REGISTRY: dict[str, str] = {
    "events": "soundthing.harness.components.events:COMPONENT",
    "environment": "soundthing.harness.components.environment:COMPONENT",
    "captions": "soundthing.harness.components.captions:COMPONENT",
    "pipeline": "soundthing.harness.components.pipeline:COMPONENT",
}

DEFAULT = "events"


def names() -> list[str]:
    return list(REGISTRY)


def get(name: str | None = None) -> Component:
    name = name or DEFAULT
    if name not in REGISTRY:
        raise KeyError(f"Unknown component {name!r}; known: {', '.join(REGISTRY)}")
    module, attr = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), attr)


__all__ = ["Clip", "Component", "Evaluation", "DEFAULT", "REGISTRY", "get", "is_config_key", "names"]
