# tif360_protein_folding

Minimal scaffold for Phases 1–2 (Transformer + isolated NeRF postprocessing).

Implemented scope status

- Phase 1 (implemented): SidechainNet-aware ingestion, dynamic length bucketing, Transformer encoder with absolute sinusoidal positional encoding, trig+distance output head, and trigonometric + distance loss training.
- Phase 2 (implemented): isolated post-processing reconstruction from internal coordinates to Cartesian coordinates, export to `.pdb` and `.gltf`, and lever-arm diagnostics logging.

Quickstart

1. Create and activate your Python virtualenv at `~/grahprnn_env`:

```bash
python -m venv ~/grahprnn_env
source ~/grahprnn_env/bin/activate
python -m pip install -r requirements.txt
```

2. Run a quick smoke run (prints the loaded config):

```bash
./scripts/run_train.sh
```

3. Run inference + postprocessing export:

```bash
./scripts/run_infer.sh
```

Config files live in `configs/` and are YAML files meant to be versioned. Use `--config` to select one, or set `CONFIG_NAME` env var to a filename under `configs/`.

Notes

- Keep entrypoints minimal: only `--config` is exposed.
- `CONFIG_NAME` can point to a config in `configs/` (with or without `.yaml`).
- Postprocessing is isolated in `src/postproc/` and runs only during inference.

Spec alignment notes

- `REQ-DI-1.01..1.05`: implemented in `src/data/dataset_full.py` via SidechainNet-aware loading, dynamic batching, and C-alpha internal target computation.
- `REQ-NN-1.01..1.04`: implemented in `src/models/transformer.py` and `src/models/heads.py`.
- `REQ-LF-1.01..1.04`: implemented in `src/models/heads.py` and `src/losses/torch_trig_loss.py`.
- `REQ-PP-2.01..2.05`: implemented in `src/postproc/nerf_runner.py`, `src/postproc/exporters.py`, `src/postproc/diagnostics.py`, and orchestrated by `src/infer.py`.

Backend notes

- Phase 2 uses `mp-nerf` by default (`postproc.nerf_impl: mpnerf`) and falls back to sequential NeRF if unavailable.
