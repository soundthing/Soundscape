# Full sound vocabulary and hierarchy

PANNs still runs the same CNN14 checkpoint. Each of its 527 output labels now has
an independent caption class, for 527 detectable classes. Event CLAP, its prompts,
and the five unmapped classes have been removed. A class still needs its own
threshold and minimum event duration. Speech and music classes are included in this expanded vocabulary.

Retained ids stay attached to their primary AudioSet label. Schema version 2
requires regenerated runs and review of old hand labels; see
[the migration notes](interfaces.md#schema-version-2-panns-only-sound-events). The 26 former alias mappings are split into independent
classes. For example, `dog_bark` now uses `Bark`, while `Dog` has its own broader
class. Otherwise a high generic dog score could incorrectly force a barking caption.
The expansion does not calibrate the new classes' thresholds or prove their accuracy.

The hierarchy is derived from [Google's AudioSet ontology](https://github.com/audioset/ontology),
which is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Attribution: Audio Set Ontology, Dan Ellis / Google, 2017. The derived parent
relationships in `config/vocabulary.yaml` are distributed under that same license.
PANNs label names/ids follow Google's official
[class label index](https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv).

To generate a separate proposal from downloaded metadata:

```bash
.venv/bin/python -m soundthing.events.vocabulary \
  --vocabulary config/vocabulary.yaml \
  --ontology data/audioset/ontology.json \
  --labels ~/panns_data/class_labels_indices.csv \
  --output /tmp/vocabulary-proposal.yaml
```

The optional `parents` lists support multiple inheritance. Unscored intermediate
ontology nodes are traversed rather than mistaken for model outputs. Existing
coarse groups also serve as fallback ancestors. Configuration rejects unknown
references, duplicate parents, and cycles.

Thresholds are evaluated independently. A qualified child suppresses its ancestors
only during their overlapping intervals. Parent-only intervals, sibling classes,
and unrelated sounds remain eligible. Caption readability padding uses the same
rule. A parent and child could represent separate sources; this classification
output cannot distinguish that case, so suppression is a display simplification.
