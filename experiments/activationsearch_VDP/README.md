# activationsearch_VDP

Compares activation functions for signed-profile PDAP on the VDP value-sample
dataset. The experiment sweeps activation, loss, gamma, and seed; each config
point writes a Run Record and a full fit-result artifact.

Canonical command:

```bash
make activationsearch_VDP
```

Executable defaults live in `conf/experiment/activationsearch_VDP.yaml`.
