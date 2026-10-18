from copy import deepcopy

from soundthing.captions import events_to_cues
from soundthing.config import SoundClass
from soundthing.contracts import CaptionEvent
from soundthing.events.hierarchy import suppress_ancestors


GRAPH = {'animal': frozenset(), 'dog': frozenset({'animal'}),
         'bark': frozenset({'animal', 'dog'}), 'bird': frozenset({'animal'})}


def event(cid, start=0, end=4):
    return CaptionEvent(start, end, cid, cid, .8, ['right'])


def spans(events):
    return {(e.class_id, e.start, e.end) for e in events}


def test_descendant_chain_suppresses_only_overlapping_intervals():
    events = [event('animal', 0, 8), event('dog', 2, 4), event('bark', 2, 3), event('siren', 2, 5)]
    original = deepcopy(events)
    result = suppress_ancestors(events, GRAPH)
    assert spans(result) == {('animal',0,2), ('animal',4,8), ('dog',3,4), ('bark',2,3), ('siren',2,5)}
    assert events == original
    for e in result:
        e.validate()
        assert e.confidence == .8 and e.modifiers == ['right']


def test_siblings_unrelated_and_nonoverlapping_events_survive():
    events = [event('dog'), event('bird'), event('siren'), event('animal', 4, 6)]
    assert spans(suppress_ancestors(events, GRAPH)) == spans(events)
    assert spans(suppress_ancestors([event('bark')], GRAPH)) == {('bark',0,4)}
    assert spans(suppress_ancestors([event('animal')], GRAPH)) == {('animal',0,4)}


def test_multiple_parents_and_overlapping_children():
    graph = {**GRAPH, 'voice': frozenset(), 'bark': frozenset({'animal','dog','voice'})}
    result = suppress_ancestors([event('animal',0,8), event('voice',0,8),
                                 event('bark',1,3),event('bark',2,5),event('bird',4,6)],graph)
    assert spans(result) == {('animal',0,1),('animal',6,8),('voice',0,1),('voice',5,8),
                             ('bark',1,3),('bark',2,5),('bird',4,6)}


def test_caption_padding_cannot_resurrect_parent_duplicates():
    events = [event('animal',0,.2),event('bark',.2,.5),event('animal',.5,1.5)]
    cues = events_to_cues(events, min_cue=1, hierarchy=GRAPH)
    assert [(c.start,c.end,c.text) for c in cues] == [(0,.2,'[animal, right]'),(.2,1.2,'[bark, right]'),(1.2,1.5,'[animal, right]')]


def test_detector_filters_ancestors_with_independent_thresholds(cfg, monkeypatch, tmp_path, inject_scores):
    import numpy as np
    import soundfile as sf
    from soundthing.events import detector
    from soundthing.events.types import FrameScores
    from soundthing.contracts import ExtractedInputs
    copy = deepcopy(cfg)
    copy.classes = [SoundClass('parent','parent',audioset=['Animal']),
                    SoundClass('child','child',audioset=['Bark'],parents=['parent']),
                    SoundClass('sibling','sibling',audioset=['Bird'],parents=['parent'])]
    copy.pipeline['events']['postprocess']['coarse_margin'] = 0
    copy.pipeline['events']['modifiers']['enabled'] = False
    audio=tmp_path/'audio.wav'; sf.write(audio,np.zeros(32000),16000)
    inputs=ExtractedInputs('clip','video',str(audio),str(tmp_path),1,2)
    monkeypatch.setattr(detector,'get_models',lambda cfg:None)
    detector._clear_caches()
    for row, expected in [([.9,.8,.7],{'child','sibling'}),([.9,.2,.1],{'parent'}),([.1,.8,.1],{'child'})]:
        detector._clear_caches()
        inject_scores(lambda:FrameScores(np.array([.5,1.]),copy.class_ids,np.tile(row,(2,1)),.5))
        assert {e.class_id for e in detector.detect(inputs,None,copy)} == expected
    detector._clear_caches()
