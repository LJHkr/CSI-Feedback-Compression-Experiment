from pathlib import Path
import csv
import numpy as np

# Compare retained dimensions and bit precision on a small channel model.
N = 16
N_TRAIN, N_VAL, N_TEST = 8000, 2000, 5000
TRAIN_RHO = 0.85
TEST_RHOS = [0.0, 0.5, 0.85]
BUDGETS = [32, 64, 96, 128]
K_VALUES = [2, 4, 6, 8, 12, 16]
MAX_BITS = 16                   # At most 16 bits per real or imaginary axis
SNR_DB = 10
N_SEEDS = 10
RANGE_FACTORS = [1.5, 2, 2.5, 3, 3.5, 4]
OUT = Path(__file__).resolve().parent / 'results'


def channel(rng, n, rho):
    i = np.arange(N)
    covariance = rho ** np.abs(i[:, None] - i[None, :])
    covariance = covariance * np.exp(0.65j * (i[:, None] - i[None, :]))
    z = (rng.normal(size=(n, N)) + 1j * rng.normal(size=(n, N))) / np.sqrt(2)
    return z @ np.linalg.cholesky(covariance).T


def quantize_one(x, b, a):
    if b == 0:
        return np.zeros_like(x)
    levels = 2 ** int(b)
    step = 2 * a / levels
    # Saturate at the outermost level; send one index for each axis.
    real_index = np.clip(np.floor((x.real + a) / step), 0, levels - 1)
    imag_index = np.clip(np.floor((x.imag + a) / step), 0, levels - 1)
    return (-a + (real_index + 0.5) * step
            + 1j * (-a + (imag_index + 0.5) * step))


def fit_quantizers(values):
    ranges = np.zeros((N, MAX_BITS + 1))
    errors = np.zeros_like(ranges)
    for j in range(N):
        x = values[:, j]
        sigma = np.sqrt(np.mean(np.abs(x) ** 2) / 2)
        errors[j, 0] = np.mean(np.abs(x) ** 2)
        for b in range(1, MAX_BITS + 1):
            losses = [np.mean(np.abs(x - quantize_one(x, b, f * sigma)) ** 2)
                      for f in RANGE_FACTORS]
            best = np.argmin(losses)
            ranges[j, b] = RANGE_FACTORS[best] * sigma
            errors[j, b] = losses[best]
    return ranges, errors


def equal_bits(budget, k):
    b = np.zeros(N, dtype=int)
    q, remainder = divmod(budget // 2, k)
    b[:k] = q
    b[:remainder] += 1
    return b


def unequal_bits(budget, k, errors):
    b = np.zeros(N, dtype=int)
    for _ in range(budget // 2):
        gain = np.full(k, -np.inf)
        for j in range(k):
            if b[j] < MAX_BITS:
                gain[j] = errors[j, b[j]] - errors[j, b[j] + 1]
        b[np.argmax(gain)] += 1
    return b


def reconstruct(h, mean, basis, bits, limits):
    coefficients = (h - mean) @ basis.conj()
    q = np.column_stack([quantize_one(coefficients[:, j], bits[j], limits[j])
                         for j in range(N)])
    return q @ basis.T + mean


def nmse(h, h_hat):
    return np.mean(np.sum(np.abs(h - h_hat) ** 2, axis=1)
                   / np.sum(np.abs(h) ** 2, axis=1))


def sharing_cost(method, bits):
    active = np.count_nonzero(bits)
    # Count the mean, ranges and basis as uncompressed float64 values.
    # Share 5-bit allocations and a 5-bit count once; the uplink payload stays B bits.
    if method == 'full_uniform':
        return 64 * (2 * N + N)
    return 64 * (2 * N + 2 * N * active + active) + 5 * (active + 1)


def evaluate(h, h_hat, gains, ideal_beam, ideal_rate, codebook):
    chosen = np.argmax(np.abs(h_hat.conj() @ codebook) ** 2, axis=1)
    rate = np.log2(1 + 10 ** (SNR_DB / 10) * gains[np.arange(len(h)), chosen])
    error = nmse(h, h_hat)
    return dict(nmse=error, nmse_db=10 * np.log10(error), rate=np.mean(rate),
                rate_loss=np.mean(ideal_rate - rate),
                beam_agreement=np.mean(chosen == ideal_beam))


def save_rows(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    runs, dimensions, allocations, eigenvalues, objectives = [], [], [], [], []
    # Use the same 64 unit-norm DFT beams for every method.
    codebook = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(64)) / 64) / np.sqrt(N)

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(2026 + seed)
        train = channel(rng, N_TRAIN, TRAIN_RHO)
        val = channel(rng, N_VAL, TRAIN_RHO)
        mean = train.mean(axis=0)
        centered = train - mean
        covariance = centered.T @ centered.conj() / len(train)
        eigenvalue, basis = np.linalg.eigh(covariance)
        basis = basis[:, ::-1]
        eigenvalue = eigenvalue[::-1]
        pca_values = centered @ basis.conj()
        pca_ranges, pca_errors = fit_quantizers(pca_values)
        full_ranges, _ = fit_quantizers(centered)
        for j, value in enumerate(eigenvalue):
            eigenvalues.append(dict(seed=seed, component=j + 1, eigenvalue=value))

        val_gains = np.abs(val.conj() @ codebook) ** 2
        val_beam = np.argmax(val_gains, axis=1)
        val_rate = np.log2(1 + 10 ** (SNR_DB / 10) * val_gains[np.arange(N_VAL), val_beam])
        models, trials, rate_models = {}, {}, {}
        for budget in BUDGETS:
            bits = equal_bits(budget, N)
            limits = full_ranges[np.arange(N), bits]
            models[budget, 'full_uniform'] = dict(
                basis=np.eye(N), bits=bits, limits=limits, k=N,
                validation_nmse=nmse(val, reconstruct(val, mean, np.eye(N), bits, limits)))

            for method in ['pca_equal', 'pca_unequal']:
                best_error, best_rate_loss = np.inf, np.inf
                for k in K_VALUES:
                    if budget > 2 * k * MAX_BITS:
                        continue                  # This dimension cannot hold the budget at the per-axis limit.
                    bits = (equal_bits(budget, k) if method == 'pca_equal'
                            else unequal_bits(budget, k, pca_errors))
                    limits = pca_ranges[np.arange(N), bits]
                    val_hat = reconstruct(val, mean, basis, bits, limits)
                    error = nmse(val, val_hat)
                    model = dict(basis=basis, bits=bits, limits=limits, k=k,
                                 validation_nmse=error)
                    trials[budget, method, k] = model
                    if budget == 64:
                        val_loss = evaluate(val, val_hat, val_gains, val_beam, val_rate, codebook)['rate_loss']
                        if val_loss < best_rate_loss:
                            best_rate_loss = val_loss
                            rate_models[method] = model
                    if error < best_error:
                        best_error = error
                        models[budget, method] = model

            for method in ['full_uniform', 'pca_equal', 'pca_unequal']:
                m = models[budget, method]
                allocations.append(dict(
                    seed=seed, budget=budget, method=method, k=m['k'],
                    active_k=np.count_nonzero(m['bits']),
                    real_bits=' '.join(map(str, m['bits'])),
                    validation_nmse=m['validation_nmse'],
                    sharing_bits=sharing_cost(method, m['bits'])))

        for rho in TEST_RHOS:
            test = channel(rng, N_TEST, rho)
            gains = np.abs(test.conj() @ codebook) ** 2
            ideal_beam = np.argmax(gains, axis=1)
            ideal_rate = np.log2(1 + 10 ** (SNR_DB / 10) * gains[np.arange(N_TEST), ideal_beam])
            for (budget, method), m in models.items():
                h_hat = reconstruct(test, mean, m['basis'], m['bits'], m['limits'])
                row = dict(seed=seed, train_rho=TRAIN_RHO, test_rho=rho,
                           budget=budget, method=method, k=m['k'],
                           active_k=np.count_nonzero(m['bits']))
                row.update(evaluate(test, h_hat, gains, ideal_beam, ideal_rate, codebook))
                row['sharing_bits'] = sharing_cost(method, m['bits'])
                runs.append(row)

            # Compare choosing k by validation NMSE and by validation rate loss.
            for method in ['pca_equal', 'pca_unequal']:
                for selector, m in [('nmse', models[64, method]), ('rate', rate_models[method])]:
                    h_hat = reconstruct(test, mean, basis, m['bits'], m['limits'])
                    row = dict(seed=seed, train_rho=TRAIN_RHO, test_rho=rho,
                               budget=64, method=method, selector=selector, k=m['k'],
                               active_k=np.count_nonzero(m['bits']))
                    row.update(evaluate(test, h_hat, gains, ideal_beam, ideal_rate, codebook))
                    row['sharing_bits'] = sharing_cost(method, m['bits'])
                    objectives.append(row)

            if rho == TRAIN_RHO:
                for (budget, method, k), m in trials.items():
                    if budget != 64:
                        continue
                    h_hat = reconstruct(test, mean, basis, m['bits'], m['limits'])
                    row = dict(seed=seed, train_rho=TRAIN_RHO, test_rho=rho,
                               budget=budget, method=method, k=k,
                               active_k=np.count_nonzero(m['bits']))
                    row.update(evaluate(test, h_hat, gains, ideal_beam, ideal_rate, codebook))
                    row['sharing_bits'] = sharing_cost(method, m['bits'])
                    dimensions.append(row)
        print(f'Seed {seed + 1}/{N_SEEDS} finished', flush=True)

    save_rows('run_results.csv', runs)
    save_rows('dimension_results.csv', dimensions)
    save_rows('allocations.csv', allocations)
    save_rows('eigenvalues.csv', eigenvalues)
    save_rows('objective_results.csv', objectives)
    print('Results saved in experiment/results', flush=True)
