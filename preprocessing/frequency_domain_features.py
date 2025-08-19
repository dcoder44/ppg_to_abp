import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def extract_frequency_domain_features(ppg_signal, fs=125):
    """
    Extracts frequency-domain features from PPG signal.

    Args:
        ppg_signal (array): Raw PPG signal.
        fs (int): Sampling frequency in Hz (default = 125Hz).

    Returns:
        dict: Dictionary containing extracted frequency-domain features.
    """
    features = {}

    # ** 1. Compute Power Spectral Density (PSD) using Welch’s Method **
    freqs, psd = signal.welch(ppg_signal, fs=fs, nperseg=fs*4)  # 4-second windows

    # ** 2. Compute Energy in Low-Frequency (LF) & High-Frequency (HF) Bands **
    lf_band = (0.1, 15)  # LF: 0.04–0.15 Hz
    hf_band = (15, 60)   # HF: 0.15–0.4 Hz

    lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])

    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])  # Energy in LF band
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])  # Energy in HF band

    # ** 3. Find Dominant Frequency & Its Magnitude **
    dominant_idx = np.argmax(psd)  # Index of max PSD
    dominant_freq = freqs[dominant_idx]
    dominant_magnitude = psd[dominant_idx]

    # ** 4. Compute LF/HF Ratio **
    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else np.nan

    # ** Store Features **
    features["LF_Power"] = lf_power
    features["HF_Power"] = hf_power
    features["LF_HF_Ratio"] = lf_hf_ratio
    features["Dominant_Frequency"] = dominant_freq
    features["Dominant_Frequency_Magnitude"] = dominant_magnitude

    return features
