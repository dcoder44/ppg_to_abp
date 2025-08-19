import numpy as np
from scipy.signal import butter, filtfilt, decimate, resample, find_peaks


def is_clipped_gradient(signal, gradient_threshold=1e-5, flat_segment_length=10):
    gradient = np.abs(np.diff(signal))
    flat_segments = gradient < gradient_threshold

    # Check for consecutive flat segments indicating clipping
    consecutive_flat = np.split(flat_segments, np.where(np.diff(flat_segments) != 0)[0]+1)
    return any(len(segment) >= flat_segment_length for segment in consecutive_flat if segment[0])


def auto_correlation_peaks_mean(x, signal_type):
    if signal_type == "abp" or signal_type == "ecg":
        x = x - np.mean(x)
    correlation = np.correlate(x, x, mode='full')
    correlation = correlation[len(x) - 1:]
    correlation_normalized = correlation / correlation[0]
    peaks = find_peaks(correlation)
    peaks_sum = 0
    peaks_mean = 0
    if peaks[0].__len__() >= 4:
        for peak in peaks[0]:
            peaks_sum = peaks_sum + correlation_normalized[peak]
        peaks_mean = peaks_sum / peaks[0].__len__()
    return peaks_mean


def is_record_good(signal, signal_type):
    return auto_correlation_peaks_mean(signal, signal_type) > 0.04 and not is_clipped_gradient(signal)
