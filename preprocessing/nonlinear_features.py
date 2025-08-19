import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import antropy as ant  # For Sample Entropy & DFA
import nolds  # For Hurst Exponent & Fractal Dimension


def sample_entropy(signal, m=2, r=0.2):
    """
    Computes Sample Entropy (SampEn) for a signal.

    Args:
        signal (array): Input PPG signal.
        m (int): Embedding dimension.
        r (float): Tolerance threshold (0.2 * standard deviation of signal).

    Returns:
        float: Sample Entropy value.
    """
    return ant.sample_entropy(signal, order=m)


def shannon_entropy(signal):
    """
    Computes Shannon Entropy of a signal.

    Args:
        signal (array): Input PPG signal.

    Returns:
        float: Shannon Entropy value.
    """
    prob_dist = np.histogram(signal, bins=100, density=True)[0]
    prob_dist = prob_dist[prob_dist > 0]  # Remove zero probabilities
    return -np.sum(prob_dist * np.log2(prob_dist))


def extract_nonlinear_features(ppg_signal):
    """
    Extracts nonlinear features from a PPG signal.

    Args:
        ppg_signal (array): Raw PPG signal.

    Returns:
        dict: Dictionary containing extracted nonlinear features.
    """
    features = {}

    # ** 1. Sample Entropy (SampEn) **
    features["Sample_Entropy"] = sample_entropy(ppg_signal)

    # ** 2. Detrended Fluctuation Analysis (DFA) **
    features["DFA_Exponent"] = ant.detrended_fluctuation(ppg_signal)

    # ** 3. Fractal Dimension (Higuchi) **
    features["Fractal_Dimension"] = nolds.hurst_rs(ppg_signal)

    # ** 4. Shannon Entropy **
    features["Shannon_Entropy"] = shannon_entropy(ppg_signal)

    # ** 5. Hurst Exponent (Long-term memory/self-similarity) **
    features["Hurst_Exponent"] = nolds.hurst_rs(ppg_signal)

    return features


# ** Generate a Sample PPG Signal with Noise **
fs = 125  # Sampling frequency
t = np.linspace(0, 10, fs * 10)  # 10-second signal
ppg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 0.1, len(t))  # Simulated PPG

# ** Extract Nonlinear Features **
nonlinear_features = extract_nonlinear_features(ppg_signal)

# ** Print Extracted Features **
for key, value in nonlinear_features.items():
    print(f"{key}: {value:.4f}")
