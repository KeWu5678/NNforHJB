# penaltypowers

Compares finite-step PDAP penalty powers across the VDP and pendulum value-sample
datasets. The experiment sweeps every activation marked homogeneous in
`src.config.activations`, plus power, loss, gamma, and seed; each config point
writes a Run Record and a full fit-result artifact.

Canonical command:

```bash
make penaltypowers
```

Executable defaults live in `conf/experiment/penaltypowers.yaml`.
