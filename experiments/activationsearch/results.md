# activationsearch Results

## Sweep coverage

- Configured data: vdp, pendulum
- Observed data: vdp
- Missing configured data: pendulum
- Configured activations: tanh, softplus, matern52, gaussian, gelu_squared, silu_squared, rcip_2, snake_b0_25, lisht, gausscent_1
- Observed activations: gausscent_1, gaussian, gelu_squared, lisht, matern52, rcip_2, silu_squared, snake_b0_25, softplus, tanh
- Missing configured activations: none

Best gamma per data/kind/insertion/activation/loss/seed

| data | kind        | insertion   | activation   | loss | seed | gamma | neurons | Val L2   | Val H1   | score    |
| ---- | ----------- | ----------- | ------------ | ---- | ---- | ----- | ------- | -------- | -------- | -------- |
| vdp  | semiconcave | finite_step | gausscent_1  | h1   | 42   | 0     | 4       | 4.56e-01 | 2.68e-01 | 1.07e+00 |
| vdp  | semiconcave | finite_step | gausscent_1  | l2   | 42   | 0     | 0       | 3.40e-01 | 7.17e-01 | 7.17e-01 |
| vdp  | semiconcave | finite_step | gaussian     | h1   | 42   | 0     | 13      | 4.26e-01 | 1.17e-01 | 1.52e+00 |
| vdp  | semiconcave | finite_step | gaussian     | l2   | 42   | 0     | 0       | 3.40e-01 | 7.17e-01 | 7.17e-01 |
| vdp  | semiconcave | finite_step | gelu_squared | h1   | 42   | 0     | 0       | 3.75e-01 | 6.58e-01 | 6.58e-01 |
| vdp  | semiconcave | finite_step | gelu_squared | l2   | 42   | 0     | 0       | 3.40e-01 | 7.17e-01 | 7.17e-01 |
| vdp  | semiconcave | finite_step | lisht        | h1   | 42   | 10    | 1       | 4.49e-01 | 4.43e-01 | 4.43e-01 |
| vdp  | semiconcave | finite_step | lisht        | l2   | 42   | 10    | 1       | 1.95e-01 | 6.21e-01 | 6.21e-01 |
| vdp  | semiconcave | finite_step | matern52     | h1   | 42   | 0     | 5       | 4.59e-01 | 2.55e-01 | 1.27e+00 |
| vdp  | semiconcave | finite_step | matern52     | l2   | 42   | 0     | 0       | 3.40e-01 | 7.17e-01 | 7.17e-01 |
| vdp  | semiconcave | finite_step | rcip_2       | h1   | 42   | 0     | 10      | 4.53e-01 | 2.78e-01 | 2.78e+00 |
| vdp  | semiconcave | finite_step | rcip_2       | l2   | 42   | 0     | 0       | 3.40e-01 | 7.17e-01 | 7.17e-01 |
| vdp  | semiconcave | finite_step | silu_squared | h1   | 42   | 10    | 0       | 3.75e-01 | 6.58e-01 | 6.58e-01 |
| vdp  | semiconcave | finite_step | silu_squared | l2   | 42   | 0     | 1       | 3.90e-01 | 6.83e-01 | 6.83e-01 |
| vdp  | semiconcave | finite_step | snake_b0_25  | h1   | 42   | 0     | 1       | 4.74e-01 | 6.51e-01 | 6.51e-01 |
| vdp  | semiconcave | finite_step | snake_b0_25  | l2   | 42   | 10    | 1       | 2.33e-01 | 6.50e-01 | 6.50e-01 |
| vdp  | semiconcave | finite_step | softplus     | h1   | 42   | 1     | 1       | 5.43e-01 | 3.12e-01 | 3.12e-01 |
| vdp  | semiconcave | finite_step | softplus     | l2   | 42   | 10    | 1       | 1.85e-01 | 6.17e-01 | 6.17e-01 |
| vdp  | semiconcave | finite_step | tanh         | h1   | 42   | 0     | 2       | 4.77e-01 | 2.35e-01 | 4.71e-01 |
| vdp  | semiconcave | finite_step | tanh         | l2   | 42   | 0     | 0       | 3.40e-01 | 7.17e-01 | 7.17e-01 |
| vdp  | semiconcave | profile     | gausscent_1  | h1   | 42   | 10    | 9       | 4.49e-01 | 2.11e-01 | 1.90e+00 |
| vdp  | semiconcave | profile     | gausscent_1  | l2   | 42   | 1     | 1       | 2.88e-01 | 6.88e-01 | 6.88e-01 |
| vdp  | semiconcave | profile     | gaussian     | h1   | 42   | 0     | 15      | 4.19e-01 | 1.12e-01 | 1.67e+00 |
| vdp  | semiconcave | profile     | gaussian     | l2   | 42   | 10    | 1       | 2.49e-01 | 7.03e-01 | 7.03e-01 |
| vdp  | semiconcave | profile     | gelu_squared | h1   | 42   | 0.1   | 18      | 4.26e-01 | 1.20e-01 | 2.16e+00 |
| vdp  | semiconcave | profile     | gelu_squared | l2   | 42   | 1     | 5       | 8.49e-02 | 5.60e-01 | 2.80e+00 |
| vdp  | semiconcave | profile     | lisht        | h1   | 42   | 0     | 4       | 4.84e-01 | 2.23e-01 | 8.91e-01 |
| vdp  | semiconcave | profile     | lisht        | l2   | 42   | 10    | 3       | 1.45e-01 | 5.81e-01 | 1.74e+00 |
| vdp  | semiconcave | profile     | matern52     | h1   | 42   | 1     | 17      | 4.25e-01 | 1.13e-01 | 1.93e+00 |
| vdp  | semiconcave | profile     | matern52     | l2   | 42   | 10    | 1       | 2.57e-01 | 7.05e-01 | 7.05e-01 |
| vdp  | semiconcave | profile     | rcip_2       | h1   | 42   | 0     | 19      | 4.17e-01 | 1.86e-01 | 3.54e+00 |
| vdp  | semiconcave | profile     | rcip_2       | l2   | 42   | 10    | 1       | 2.76e-01 | 7.06e-01 | 7.06e-01 |
| vdp  | semiconcave | profile     | silu_squared | h1   | 42   | 0.1   | 14      | 4.40e-01 | 1.44e-01 | 2.02e+00 |
| vdp  | semiconcave | profile     | silu_squared | l2   | 42   | 1     | 4       | 1.16e-01 | 5.60e-01 | 2.24e+00 |
| vdp  | semiconcave | profile     | snake_b0_25  | h1   | 42   | 0.1   | 3       | 5.57e-01 | 2.69e-01 | 8.07e-01 |
| vdp  | semiconcave | profile     | snake_b0_25  | l2   | 42   | 10    | 1       | 1.73e-01 | 5.98e-01 | 5.98e-01 |
| vdp  | semiconcave | profile     | softplus     | h1   | 42   | 0.1   | 2       | 5.61e-01 | 2.75e-01 | 5.50e-01 |
| vdp  | semiconcave | profile     | softplus     | l2   | 42   | 1     | 2       | 1.10e-01 | 5.57e-01 | 1.11e+00 |
| vdp  | semiconcave | profile     | tanh         | h1   | 42   | 0.1   | 29      | 4.46e-01 | 1.40e-01 | 4.07e+00 |
| vdp  | semiconcave | profile     | tanh         | l2   | 42   | 0     | 4       | 2.41e-01 | 6.90e-01 | 2.76e+00 |
| vdp  | signed      | finite_step | gausscent_1  | h1   | 42   | 0     | 15      | 4.70e-01 | 2.66e-01 | 3.99e+00 |
| vdp  | signed      | finite_step | gausscent_1  | l2   | 42   | 0     | 3       | 3.00e-01 | 8.35e-01 | 2.50e+00 |
| vdp  | signed      | finite_step | gaussian     | h1   | 42   | 0     | 12      | 4.61e-01 | 2.34e-01 | 2.80e+00 |
| vdp  | signed      | finite_step | gaussian     | l2   | 42   | 0.1   | 3       | 4.07e-01 | 8.93e-01 | 2.68e+00 |
| vdp  | signed      | finite_step | gelu_squared | h1   | 42   | 1     | 28      | 3.98e-01 | 1.43e-01 | 4.00e+00 |
| vdp  | signed      | finite_step | gelu_squared | l2   | 42   | 0.1   | 7       | 1.09e-01 | 5.84e-01 | 4.09e+00 |
| vdp  | signed      | finite_step | lisht        | h1   | 42   | 1     | 19      | 4.82e-01 | 2.46e-01 | 4.67e+00 |
| vdp  | signed      | finite_step | lisht        | l2   | 42   | 0     | 4       | 3.67e-01 | 7.57e-01 | 3.03e+00 |
| vdp  | signed      | finite_step | matern52     | h1   | 42   | 0     | 12      | 4.57e-01 | 2.38e-01 | 2.85e+00 |
| vdp  | signed      | finite_step | matern52     | l2   | 42   | 0.1   | 3       | 4.29e-01 | 9.08e-01 | 2.72e+00 |
| vdp  | signed      | finite_step | rcip_2       | h1   | 42   | 1     | 43      | 4.22e-01 | 1.64e-01 | 7.06e+00 |
| vdp  | signed      | finite_step | rcip_2       | l2   | 42   | 0     | 3       | 4.97e-01 | 9.40e-01 | 2.82e+00 |
| vdp  | signed      | finite_step | silu_squared | h1   | 42   | 1     | 27      | 4.64e-01 | 1.21e-01 | 3.27e+00 |
| vdp  | signed      | finite_step | silu_squared | l2   | 42   | 0     | 6       | 3.37e-01 | 6.89e-01 | 4.13e+00 |
| vdp  | signed      | finite_step | snake_b0_25  | h1   | 42   | 0.1   | 18      | 4.32e-01 | 1.49e-01 | 2.69e+00 |
| vdp  | signed      | finite_step | snake_b0_25  | l2   | 42   | 1     | 7       | 2.08e-01 | 7.12e-01 | 4.99e+00 |
| vdp  | signed      | finite_step | softplus     | h1   | 42   | 1     | 11      | 5.83e-01 | 3.05e-01 | 3.36e+00 |
| vdp  | signed      | finite_step | softplus     | l2   | 42   | 0.1   | 5       | 1.98e-01 | 6.85e-01 | 3.42e+00 |
| vdp  | signed      | finite_step | tanh         | h1   | 42   | 1     | 19      | 5.21e-01 | 4.84e-01 | 9.20e+00 |
| vdp  | signed      | finite_step | tanh         | l2   | 42   | 0.1   | 6       | 4.58e-01 | 9.54e-01 | 5.72e+00 |
| vdp  | signed      | profile     | gausscent_1  | h1   | 42   | 0     | 18      | 4.54e-01 | 2.49e-01 | 4.48e+00 |
| vdp  | signed      | profile     | gausscent_1  | l2   | 42   | 0     | 3       | 3.00e-01 | 8.35e-01 | 2.50e+00 |
| vdp  | signed      | profile     | gaussian     | h1   | 42   | 0     | 12      | 4.60e-01 | 2.31e-01 | 2.78e+00 |
| vdp  | signed      | profile     | gaussian     | l2   | 42   | 10    | 2       | 3.77e-01 | 8.79e-01 | 1.76e+00 |
| vdp  | signed      | profile     | gelu_squared | h1   | 42   | 0     | 25      | 4.26e-01 | 1.10e-01 | 2.75e+00 |
| vdp  | signed      | profile     | gelu_squared | l2   | 42   | 10    | 7       | 1.03e-01 | 5.72e-01 | 4.01e+00 |
| vdp  | signed      | profile     | lisht        | h1   | 42   | 1     | 17      | 5.39e-01 | 2.61e-01 | 4.44e+00 |
| vdp  | signed      | profile     | lisht        | l2   | 42   | 0     | 25      | 1.08e-01 | 6.05e-01 | 1.51e+01 |
| vdp  | signed      | profile     | matern52     | h1   | 42   | 1     | 15      | 4.80e-01 | 2.28e-01 | 3.42e+00 |
| vdp  | signed      | profile     | matern52     | l2   | 42   | 10    | 2       | 3.94e-01 | 8.91e-01 | 1.78e+00 |
| vdp  | signed      | profile     | rcip_2       | h1   | 42   | 0.1   | 27      | 4.31e-01 | 2.67e-01 | 7.20e+00 |
| vdp  | signed      | profile     | rcip_2       | l2   | 42   | 0     | 3       | 4.97e-01 | 9.40e-01 | 2.82e+00 |
| vdp  | signed      | profile     | silu_squared | h1   | 42   | 10    | 24      | 4.31e-01 | 1.27e-01 | 3.05e+00 |
| vdp  | signed      | profile     | silu_squared | l2   | 42   | 0     | 8       | 1.04e-01 | 5.73e-01 | 4.59e+00 |
| vdp  | signed      | profile     | snake_b0_25  | h1   | 42   | 10    | 19      | 4.42e-01 | 1.37e-01 | 2.60e+00 |
| vdp  | signed      | profile     | snake_b0_25  | l2   | 42   | 10    | 22      | 3.44e-02 | 4.62e-01 | 1.02e+01 |
| vdp  | signed      | profile     | softplus     | h1   | 42   | 10    | 10      | 5.90e-01 | 3.78e-01 | 3.78e+00 |
| vdp  | signed      | profile     | softplus     | l2   | 42   | 0.1   | 5       | 1.98e-01 | 6.85e-01 | 3.42e+00 |
| vdp  | signed      | profile     | tanh         | h1   | 42   | 0     | 11      | 5.15e-01 | 4.86e-01 | 5.34e+00 |
| vdp  | signed      | profile     | tanh         | l2   | 42   | 0.1   | 4       | 4.56e-01 | 9.56e-01 | 3.82e+00 |
