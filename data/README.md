# Data placement

This repository does not include SPeDaC1 or SPY data. Put the unzipped benchmark
files under `data/` before running the scripts.

Required layout:

```text
data/spedac1/train.jsonl
data/spedac1/valid.jsonl
data/spedac1/test.jsonl
data/spedac1/label_map.yaml        # optional

data/spy/train.jsonl
data/spy/valid.jsonl
data/spy/test.jsonl
data/spy/label_map.yaml            # optional
```

Each JSONL row must contain at least `text` and `label`.

```json
{"text": "...", "label": "Sensitive"}
```

The accepted labels are exactly `Sensitive` and `Non-sensitive`.
