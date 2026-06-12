# penaltypowers Results

No current run records were found under `rawdata/logs/multirun/penaltypowers`.

Run:

```bash
make penaltypowers
```

The sweep is configured for:

- data: vdp, pendulum
- homogeneous activations: relu, leaky_relu, abs_act, cubic, x_absx, quartic, smoothy_relu_sphere, leaky_relu2_a0_001_sphere, leaky_relu2_sphere, leaky_relu2_a0_015_sphere, leaky_relu2_a0_02_sphere, leaky_relu2_a0_025_sphere, leaky_relu2_a0_0375_sphere, leaky_relu2_a0_05_sphere, leaky_relu2_a0_05, leaky_relu2_a0_0625_sphere, leaky_relu2_a0_075_sphere, leaky_relu2_a0_1_sphere, relu2
- powers: 2.0, 2.01, 3.0, 4.0, 5.0
- gamma: 0, 0.01, 0.1, 1, 10
- loss weights: [1.0, 0.0], [1.0, 1.0]

After the sweep completes, `experiments/penaltypowers/analysis.py` will replace
this file with the best-gamma table.
