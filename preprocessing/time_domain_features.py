import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from scipy.integrate import simpson  # For area under the curve
import neurokit2 as nk  # For PRV (Pulse Rate Variability) features


def extract_time_domain_features(ppg_signal, fs=125):
    """
    Extracts time-domain features from PPG signal and its derivatives.

    Args:
        ppg_signal (array): Raw PPG signal.
        fs (int): Sampling frequency in Hz (default = 125Hz).

    Returns:
        dict: Dictionary containing extracted features.
    """

    features = {}

    # ** Compute First and Second Derivative (VPG and APG) **
    vpg = np.gradient(ppg_signal)  # Velocity PPG (1st derivative)
    apg = np.gradient(vpg)  # Acceleration PPG (2nd derivative)

    # ** Skewness (Measures asymmetry) **
    features["PPG_Skewness"] = stats.skew(ppg_signal)
    features["VPG_Skewness"] = stats.skew(vpg)
    features["APG_Skewness"] = stats.skew(apg)

    # ** Margin Factor (MF) **
    features["PPG_Margin_Factor"] = np.max(ppg_signal) / np.mean(ppg_signal ** 2)
    features["VPG_Margin_Factor"] = np.max(vpg) / np.mean(vpg ** 2)
    features["APG_Margin_Factor"] = np.max(apg) / np.mean(apg ** 2)

    # ** K Value (Variability of Pulse Waveform) **
    ppg_min, ppg_max = np.min(ppg_signal), np.max(ppg_signal)
    ppg_mean = np.mean(ppg_signal)
    features["PPG_K_Value"] = (ppg_mean - ppg_min) / (ppg_max - ppg_min)

    # ** Pulse Width, Rise Time, and Decay Time **
    peaks, _ = signal.find_peaks(ppg_signal, distance=fs//2)  # Find systolic peaks
    if len(peaks) > 1:
        pulse_widths = np.diff(peaks) / fs  # Time between peaks
        features["Mean_Pulse_Width"] = np.mean(pulse_widths)
        features["Pulse_Width_Std"] = np.std(pulse_widths)

        rise_times = [np.argmax(ppg_signal[p:]) / fs for p in peaks[:-1]]  # Rise time
        decay_times = [np.argmax(ppg_signal[p:][::-1]) / fs for p in peaks[:-1]]  # Decay time
        features["Mean_Rise_Time"] = np.mean(rise_times)
        features["Mean_Decay_Time"] = np.mean(decay_times)
    else:
        features["Mean_Pulse_Width"] = np.nan
        features["Pulse_Width_Std"] = np.nan
        features["Mean_Rise_Time"] = np.nan
        features["Mean_Decay_Time"] = np.nan

    # ** Pulse Rate Variability (PRV) using PPG Peak Indices **
    if len(peaks) > 1:
        hrv_features = nk.hrv_time({"PPG_Peaks": peaks}, sampling_rate=fs)
        features["PPG_PulseRate"] = 60 / np.mean(np.diff(peaks) / fs)  # Convert to BPM
        features["PPG_PulseRate_Variability"] = hrv_features["HRV_SDNN"][0] if "HRV_SDNN" in hrv_features else np.nan
    else:
        features["PPG_PulseRate"] = np.nan
        features["PPG_PulseRate_Variability"] = np.nan

    # ** Area Under Curve (AUC) - Measures waveform energy **
    features["PPG_AUC"] = simpson(ppg_signal, dx=1/fs)
    features["VPG_AUC"] = simpson(vpg, dx=1/fs)
    features["APG_AUC"] = simpson(apg, dx=1/fs)

    return features
