from time_domain_features import extract_time_domain_features
from frequency_domain_features import extract_frequency_domain_features
from ecg_features import extract_ecg_features
import numpy as np
from scipy.signal import find_peaks
from nonlinear_features import extract_nonlinear_features


def calculate_heart_rate(ppg, fs):
    peaks, _ = find_peaks(ppg, distance=fs // 4, height=0.5)  # Min distance = 0.5s
    interval = np.median(np.diff(peaks))
    heart_rate = ((1/interval)*125)*60
    return heart_rate


def extract_features(ppg, ecg, fs):
    time_domain_features = extract_time_domain_features(ppg)
    frequency_domain_features = extract_frequency_domain_features(ppg)
    non_linear_features = extract_nonlinear_features(ppg)

    heart_rate = calculate_heart_rate(ppg, fs)

    features = []

    for feature in time_domain_features.values():
        features.append(feature)
        if np.any(np.isnan(feature)):
            print("nan in time domain features")

    for feature in frequency_domain_features.values():
        features.append(feature)
        if np.any(np.isnan(feature)):
            print("nan in frequency domain features")

    for feature in non_linear_features.values():
        features.append(feature)
        if np.any(np.isnan(feature)):
            print("nan in non linear features")

    return features
