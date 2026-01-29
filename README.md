# tabular-GIA

Focused repo for the tabular gradient inversion (GIA) implementation and evaluation.

## Setup
```bash
pip install -e .[dev]
```

## Run the tabular GIA demo
```bash
python tabular_gia/main.py
```

This will load a checkpoint if present, otherwise train a model first.

To explicitly train:
```bash
python tabular_gia/train.py
```

## Configuration
- `tabular_gia/config.yaml` controls dataset path, splits, batch size, and training settings.
- Datasets live under `tabular_gia/data/...` (already included).

## Outputs
- Per-run metrics and per-row comparisons are written to `tabular_gia/results.txt`.
- Checkpoints are saved under `tabular_gia/data/<dataset>/checkpoints/`.

## Notes
- The evaluation matches reconstructed rows to originals using Hungarian assignment before scoring.
- Numerical features are z-score normalized; categoricals are one-hot encoded.

## Restoring upstream LeakPro
This repo intentionally prunes unrelated LeakPro components to keep the tabular GIA case small and reproducible.
If you want the full LeakPro functionality later, you have two options:

1) Replace the `leakpro/` folder with a fresh copy from the upstream LeakPro repo.
2) Add upstream LeakPro as a submodule (or dependency) and adjust imports accordingly.
