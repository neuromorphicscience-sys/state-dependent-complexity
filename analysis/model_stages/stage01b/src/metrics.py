import numpy as np
from scipy.signal import periodogram


def compute_metrics(pop_rate_hz, spike_counts, duration_s, bin_ms):
    x = np.asarray(pop_rate_hz, dtype=np.float64)
    if len(x) < 8:
        return {}

    mean_rate = float(np.mean(x))
    std_rate = float(np.std(x))
    synchrony_proxy = float(std_rate / (mean_rate + 1e-8))

    fs = 1000.0 / float(bin_ms)
    f, pxx = periodogram(x - np.mean(x), fs=fs)

    valid = (f >= 0.5) & (f <= min(100.0, fs / 2))
    if np.any(valid) and np.sum(pxx[valid]) > 0:
        fv = f[valid]
        pv = pxx[valid]
        idx = int(np.argmax(pv))
        dominant_frequency_hz = float(fv[idx])
        spectral_concentration = float(pv[idx] / (np.sum(pv) + 1e-12))
        pnorm = pv / (np.sum(pv) + 1e-12)
        spectral_entropy = float(
            -np.sum(pnorm * np.log(pnorm + 1e-12)) / np.log(len(pnorm))
        )
    else:
        dominant_frequency_hz = 0.0
        spectral_concentration = 0.0
        spectral_entropy = 1.0

    counts = np.asarray(spike_counts, dtype=np.float64)
    neuron_rates = counts / max(duration_s, 1e-9)
    silent_fraction = float(np.mean(neuron_rates < 0.1))
    active = neuron_rates[neuron_rates >= 0.1]
    rate_cv_across_neurons = (
        float(np.std(active) / (np.mean(active) + 1e-8))
        if len(active) else 0.0
    )

    activity_gate = np.tanh(mean_rate / 2.0)
    rhythm_score = float(
        activity_gate
        * spectral_concentration
        * (1.0 - min(spectral_entropy, 1.0))
        * np.tanh(synchrony_proxy)
        * (1.0 - silent_fraction)
    )

    return {
        "mean_rate_hz": mean_rate,
        "rate_std_hz": std_rate,
        "synchrony_proxy": synchrony_proxy,
        "dominant_frequency_hz": dominant_frequency_hz,
        "spectral_concentration": spectral_concentration,
        "spectral_entropy": spectral_entropy,
        "silent_fraction": silent_fraction,
        "rate_cv_across_neurons": rate_cv_across_neurons,
        "rhythm_score": rhythm_score,
    }


def classify_frequency_state(freq_hz, rhythm_score, spectral_concentration):
    """
    Coarse regime classification for Stage 1B.
    Thresholds are descriptive labels, not biological claims.
    """
    if rhythm_score < 0.03 or spectral_concentration < 0.10:
        return "weak_or_irregular"
    if freq_hz < 4.0:
        return "slow"
    if freq_hz < 15.0:
        return "low_frequency"
    if freq_hz < 30.0:
        return "intermediate"
    if freq_hz < 55.0:
        return "high_intermediate"
    return "high_frequency"
