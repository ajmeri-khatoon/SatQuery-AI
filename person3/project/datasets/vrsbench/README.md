# VRSBench Access

VRSBench is the remote-sensing vision-language benchmark by Xiang Li, Jian
Ding, and Mohamed Elhoseiny. The official benchmark covers image captioning,
visual grounding, and visual question answering.

Official sources:

- Project page: https://vrsbench.github.io/
- Code and documentation: https://github.com/lx709/VRSBench
- Dataset: https://huggingface.co/datasets/xiang709/VRSBench

The current Hugging Face access call is:

```python
from datasets import load_dataset
load_dataset("xiang709/VRSBench", split="train", streaming=True)
```

The older official example included `name="VRSBench"`; current `datasets`
metadata exposes only the `default` configuration, so that argument must be
omitted.

The current repository metadata publishes `train` and `val` archive-backed
splits. It does not publish a `test` archive.

The published dataset contains annotation archives such as
`Annotations_train.zip` and image archives such as `Images_train.zip`. Do not
extract or download the full dataset for the initial smoke test. The loader can
stream records and materialize only the limited samples requested by the
 evaluator, or read an existing local extraction.

## Expected Local Layout

```text
datasets/vrsbench/
├── Annotations_train/
│   └── *.json
├── Images_train/
│   └── image files
└── # corresponding split directories when available
```

Annotation records observed in the official repository include `image`,
`caption`, `qa_pairs`, and `objects`. QA entries use `question`, `answer`, and
`type`. Referring objects use `referring_sentence`, `obj_corner`, `obj_coord`,
`obj_cls`, and `is_unique` when present. The official README states that
provided evaluation boxes are normalized to 0-100; this loader normalizes box
coordinates to 0-1 for metric calculation.

The repository currently does not contain VRSBench data. The loader therefore
fails clearly for unavailable splits or missing access instead of guessing a
schema, changing splits, or downloading the full dataset.
