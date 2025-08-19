import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import wavelet
from filters import lowpass_filter


def extract_ecg_features(ecg_signal, fs=125, heart_rate=70):
    features = {}

    # ** 1. Normalize the ECG Signal (Fix Low Amplitude) **
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # ** 2. Clean ECG Signal **
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="pantompkins1985")

    # ** 3. Detect R-peaks using `find_peaks()` **
    r_peaks, _ = find_peaks(ecg_cleaned, distance=heart_rate//4, height=0.5)  # Min distance = 0.5s
    ecg_peaks = {"ECG_R_Peaks": r_peaks}

    # ** 4. Check if Enough Peaks Are Detected **
    if len(r_peaks) < 3:
        print("Warning: Too few R-peaks detected!")
        return {key: np.nan for key in ["Mean_RR_Interval", "HRV_SDNN", "Mean_PQ_Interval",
                                        "Mean_QRS_Duration", "Mean_QT_Interval", "Mean_ST_Segment"]}

    # ** 5. Delineate ECG waves using detected R-peaks **
    _, waves = nk.ecg_delineate(ecg_cleaned, peaks=ecg_peaks, sampling_rate=fs, method="dwt")

    # ** 7. Compute HRV Features (R-R Interval) **
    rr_intervals = np.diff(np.array(r_peaks)) / fs  # Convert to seconds
    features["Mean_RR_Interval"] = np.mean(rr_intervals)
    features["HRV_SDNN"] = np.std(rr_intervals)

    # ** 8. Extract P-Q, QRS, QT, ST Intervals (Only if Waves Exist) **
    def extract_interval(wave1, wave2):
        """ Extracts interval between two detected ECG wave points. """
        if (wave1 in waves and wave2 in waves and len(waves[wave1]) > 0 and len(waves[wave2]) > 0) or \
                (wave1 in waves and wave2 == "ECG_R_Peaks" and len(waves[wave1]) > 0 and len(r_peaks) > 0):

            w1 = np.array(waves[wave1])
            w2 = r_peaks if wave2 == "ECG_R_Peaks" else np.array(waves[wave2])

            if wave2 == "ECG_R_Peaks":
                if len(w2) > len(w1):
                    if np.abs(w2[-1]-w1[-1]) < np.mean(np.diff(w2)) * 0.8:
                        w2 = w2[len(w2)-len(w1):]
                    else:
                        w2 = w2[:len(w1)]
                print(f"w1 : {w1}, w2 : {w2}")

            if np.isnan(w1[-1]):
                w1 = w1[:len(w1)-1]
            if np.isnan(w2[-1]):
                w2 = w2[:len(w2)-1]
            valid_idx = min(len(w1), len(w2))
            interval = (w2[:valid_idx] - w1[:valid_idx]) / fs
            return np.mean(interval) if len(interval) > 0 else np.nan
        else:
            print("If condition not true")
        return np.nan

    # features["Mean_PQ_Interval"] = np.abs(extract_interval("ECG_P_Onsets", "ECG_R_Peaks"))
    features["Mean_QRS_Duration"] = extract_interval("ECG_Q_Peaks", "ECG_S_Peaks")
    features["Mean_QT_Interval"] = extract_interval("ECG_Q_Peaks", "ECG_T_Offsets")
    features["Mean_ST_Segment"] = extract_interval("ECG_S_Peaks", "ECG_T_Onsets")

    return features


# ** Generate a Sample ECG Signal (Simulated) **
fs = 125  # Higher sampling rate for better peak detection
ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=fs, heart_rate=150)
ecg_denoised = wavelet.denoise(ecg_signal)
ecg_filtered = lowpass_filter(ecg_denoised, 60, 125, order=4)

# ** Extract ECG Features **
ecg_features = extract_ecg_features(ecg_filtered, fs=fs, heart_rate=150)

# ** Print Extracted Features **
for key, value in ecg_features.items():
    print(f"{key}: {value:.4f}")