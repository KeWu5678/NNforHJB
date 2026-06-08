# activationsearch_pendulum

Compares activation functions for signed-profile PDAP on the open-loop pendulum
PMP value-sample dataset. The experiment sweeps activation, loss, gamma, and
seed; each config point writes a Run Record and a full fit-result artifact.

Canonical command:

```bash
make activationsearch_pendulum
```

Executable defaults live in `conf/experiment/activationsearch_pendulum.yaml`.
