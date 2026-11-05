"""Stable Python import facade for ADA's canonical 03_PERCEPTION source tree."""

from importlib import import_module

_core = import_module("03_PERCEPTION")
for _name in _core.__all__:
    globals()[_name] = getattr(_core, _name)

_input = import_module("03_PERCEPTION.input_interface")
_text = import_module("03_PERCEPTION.text_perception")
_structured = import_module("03_PERCEPTION.structured_perception")
_fusion = import_module("03_PERCEPTION.sensory_fusion")
_validation = import_module("03_PERCEPTION.perception_validation")
_patterns = import_module("03_PERCEPTION.pattern_detection")
_visual = import_module("03_PERCEPTION.visual_perception")
_audio = import_module("03_PERCEPTION.audio_perception")
_video = import_module("03_PERCEPTION.video_perception")
_entities = import_module("03_PERCEPTION.entity_recognition")
_events = import_module("03_PERCEPTION.event_detection")

for _module in (_input, _text, _structured, _fusion, _validation, _patterns, _visual, _audio, _video, _entities, _events):
    for _name in _module.__all__:
        globals()[_name] = getattr(_module, _name)

__all__ = list(_core.__all__)
for _module in (_input, _text, _structured, _fusion, _validation, _patterns, _visual, _audio, _video, _entities, _events):
    __all__.extend(name for name in _module.__all__ if name not in __all__)
