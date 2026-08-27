import numpy as np
from scipy.signal import periodogram


def compute_metrics(pop_rate_hz, spike_counts, duration_s, bin_ms):
    x = np.asarray(pop_rate_hz, dtype=np.float64)
    mean_rate = float(np.mean(x))
    std_rate = float(np.std(x))
    synchrony_proxy = float(std_rate / (mean_rate + 1e-8))

    fs = 1000.0 / float(bin_ms)
    f, pxx = periodogram(x - np.mean(x), fs=fs)
    valid = (f >= 0.5) & (f <= min(100.0, fs / 2))
    if np.any(valid) and np.sum(pxx[valid]) > 0:
        fv, pv = f[valid], pxx[valid]
        k = int(np.argmax(pv))
        dom = float(fv[k])
        sc = float(pv[k] / (np.sum(pv) + 1e-12))
        pn = pv / (np.sum(pv) + 1e-12)
        se = float(-np.sum(pn * np.log(pn + 1e-12)) / np.log(len(pn)))
    else:
        dom, sc, se = 0.0, 0.0, 1.0

    counts = np.asarray(spike_counts, dtype=np.float64)
    rates = counts / max(duration_s, 1e-9)
    silent = float(np.mean(rates < 0.1))
    active = rates[rates >= 0.1]
    rate_cv = float(np.std(active) / (np.mean(active) + 1e-8)) if len(active) else 0.0

    rhythm = float(
        np.tanh(mean_rate / 2.0)
        * sc
        * (1.0 - min(se, 1.0))
        * np.tanh(synchrony_proxy)
        * (1.0 - silent)
    )

    return {
        "mean_rate_hz": mean_rate,
        "rate_std_hz": std_rate,
        "synchrony_proxy": synchrony_proxy,
        "dominant_frequency_hz": dom,
        "spectral_concentration": sc,
        "spectral_entropy": se,
        "silent_fraction": silent,
        "rate_cv_across_neurons": rate_cv,
        "rhythm_score": rhythm,
    }
