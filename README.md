# CSI Feedback Compression Experiment

[Chinese version](README_ZH.md)

This experiment compares the number of retained CSI components and the precision assigned to each component. `csi_feedback.py` runs the simulations, and `plot_results.py` reads the CSV files and draws the figures. Parameters are listed at the top of each script.

## Run the experiment

Run these commands from the repository root:

```powershell
python -m pip install numpy pandas matplotlib
python experiment/csi_feedback.py
python experiment/plot_results.py
```

Outputs are saved in `experiment/results/`. The simulations use NumPy; the plotting script uses pandas and Matplotlib.

## Simulation settings

| Parameter | Value |
|---|---|
| Channel dimension | 16 complex components |
| Training / validation / test samples | 8,000 / 2,000 / 5,000 per seed; 5,000 test samples for each test correlation |
| Training and validation correlation | rho = 0.85 |
| Test correlations | rho = 0, 0.5, 0.85 |
| Uplink feedback budgets | 32, 64, 96, 128 bits |
| Candidate retained dimensions | 2, 4, 6, 8, 12, 16 |
| Maximum precision per real or imaginary axis | 16 bits |
| Transmit SNR | 10 dB |
| Beam codebook | 64 fixed unit-norm oversampled DFT beams |
| Independent runs | 10 seeds, numbered 2026 to 2035 |

The channel is a zero-mean complex Gaussian vector with covariance `R[i,j] = rho**abs(i-j) * exp(0.65j*(i-j))`. Its diagonal entries are one, so the average channel energy is the same at every correlation. The value 0.65 is a fixed spatial phase slope. These are illustrative simulation settings. Channel knowledge at the receiver and feedback indices are assumed error-free, so the experiment focuses on compression and quantization error.

## Feedback schemes

- `full_uniform`: uniform scalar quantization of the real and imaginary parts of the full channel, with equal precision across components.
- `pca_equal`: complex PCA fitted to the training set, followed by nearly equal allocation across the first k components. Any remaining bit pairs are assigned to the earlier components.
- `pca_unequal`: a training quantization-MSE table is used to assign each new bit pair to the component with the largest estimated error reduction. Each step adds one bit to the real part and one to the imaginary part.

If both axes of complex component i receive b_i bits, the total payload is `B = 2*sum(b_i)`. A zero-bit coefficient is reconstructed as zero before adding back the common training mean. Quantizer levels are equally spaced. At each precision, the range is selected by training MSE from 1.5, 2, 2.5, 3, 3.5 and 4 times the estimated per-axis standard deviation.

The k = 2 configuration cannot fit 96 or 128 bits under the 16-bit per-axis limit, so it is excluded at those budgets.

The main comparison selects k using validation NMSE. The mean, PCA basis, ranges, allocation and k are fixed before testing, and the same configuration is used for all three test correlations. Within each seed, all methods share the same data and beam codebook.

A supplementary comparison at 64 bits selects k using either validation NMSE or validation beamforming-rate loss, with the other steps kept the same.

## Metrics and sharing cost

NMSE is computed as `sum(abs(h-h_hat)**2)/sum(abs(h)**2)` for each test vector and then averaged over 5,000 samples. Each run's mean NMSE is converted to dB before averaging across the ten runs.

Each method uses the reconstructed channel to choose a beam, then evaluates `log2(1 + 10*abs(h^H*w)**2)` on the true channel. Rate loss is measured against the beam chosen using true CSI from the same codebook. `beam_agreement` is the fraction of matching beam choices.

Each operating point has a fixed feedback configuration, so each report carries B index bits. Downlink sharing is counted separately: the mean, quantizer ranges and PCA basis use 64 bits per real float64 value, with five bits per PCA allocation entry and five bits for the component count. The full-dimensional scheme uses a known coordinate basis. The sharing plot divides the cost of one fixed configuration by 10, 100, 1,000 and 10,000 reports; it does not combine this cost with the fixed uplink budget.

Statistics use seeds as the independent units. Figures show means and 95% t intervals over ten runs (`t_9 = 2.262`). For paired differences, methods are subtracted within each seed before the interval is calculated.

## Result files

| File | Contents |
|---|---|
| `run_results.csv` | 360 main-comparison rows: three methods, four budgets, three test correlations and ten seeds |
| `dimension_results.csv` | Retained-dimension scan at 64 bits, 120 rows |
| `objective_results.csv` | Validation-objective comparison at 64 bits, 120 rows |
| `allocations.csv` | Selected k, number of active components, bits per axis and sharing cost |
| `eigenvalues.csv` | PCA eigenvalues from each training run |
| `summary.csv`, `paired_differences.csv` | Main and dimension-scan summaries and paired intervals |
| `objective_summary.csv`, `objective_paired.csv` | Validation-objective summaries and paired intervals |
| `budget.png/pdf` | NMSE and rate loss against feedback budget with matched correlation |
| `dimensions.png/pdf` | NMSE and rate loss against retained dimension |
| `mismatch.png/pdf` | Effect of changing test correlation while keeping the trained configuration fixed |
| `selection_objective.png/pdf` | Selecting k by NMSE or by rate loss |
| `allocation_and_sharing.png/pdf` | Example bit allocations and amortized downlink sharing cost |
