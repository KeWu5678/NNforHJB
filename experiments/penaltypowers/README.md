# penaltypowers

Compares finite-step PDAP penalty powers on the VDP value-sample dataset. The
experiment sweeps power, loss, gamma, and seed; each config point writes a Run
Record and a full fit-result artifact.

Canonical command:

```bash
uv run python experiments/penaltypowers/run.py
```

Executable defaults live in `conf/experiment/penaltypowers.yaml`.
