import pywt
import numpy as np
import matplotlib.pyplot as plt


def denoise(signal, wavelet='sym4', level=2, threshold_method='soft'):
    """
    Perform wavelet denoising on a 1D signal using soft thresholding.

    Args:
        signal (array): The raw signal (e.g., PPG).
        wavelet (str): The type of wavelet to use (default: 'sym4').
        level (int): Number of decomposition levels.
        threshold_method (str): 'soft' or 'hard' thresholding.

    Returns:
        array: The denoised signal.
    """
    # Wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise standard deviation from detail coefficients at the highest level
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust estimation of noise level

    # Universal threshold (Donoho's method)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply thresholding to detail coefficients
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode=threshold_method) for c in coeffs[1:]]

    # Reconstruct the signal
    denoised_signal = pywt.waverec(new_coeffs, wavelet)

    return denoised_signal
