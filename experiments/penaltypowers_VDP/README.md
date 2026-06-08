# penaltypowers_VDP

Compares finite-step PDAP penalty powers on the VDP value-sample dataset. The
experiment sweeps power, loss, gamma, and seed; each config point writes a Run
Record and a full fit-result artifact.

Canonical command:

```bash
make penaltypowers_VDP
```

Executable defaults live in `conf/experiment/penaltypowers_VDP.yaml`.
