import numpy as np
import h5py
import matplotlib.pyplot as plt
import wavelet
from filters import bandpass_filter, lowpass_filter
from data_validation import is_record_good
from feature_extraction import extract_features

"""
Preprocessing pipeline:
1. Remove records with length < 60000 (8 minutes) or containing nan values
2. Wavelet denoising
3. Segmentation. Segments of size 1000 (8s). Optional -> sliding window of 3s
4. Filtering
5. Normalise PPG
6. Remove any bad segments
"""

segment_size = 1000

ppg = np.zeros([1, 1000])
abp = np.zeros([1, 1000])
ecg = np.zeros([1, 1000])
features = np.zeros([1, 26])  # 26 features
total_segments = 0
final_segments = 0

for n in range(1, 5):
    # Open the .mat file
    file_path = f"../uci_database/Part_{n}.mat"
    with h5py.File(file_path, "r") as mat_file:
        print("Keys in the file:", list(mat_file.keys()))  # List the variables

        dataset_name = f"Part_{n}"  # Update with the correct dataset name
        data = mat_file[dataset_name][()]  # Read dataset

        # for index in range(0, 1):
        for index in range(0, data.shape[0]):
            print(f"{index}/{data.shape[0]}")

            ref_obj = data[index, 0]  # Get HDF5 object reference
            actual_data = mat_file[ref_obj][()]  # Dereference to get the actual data
            if actual_data.shape[0] < 60000:
                continue
            ppg_signal = actual_data[:, 0]
            abp_signal = actual_data[:, 1]
            ecg_signal = actual_data[:, 2]

            # Apply wavelet denoising
            ppg_denoised = wavelet.denoise(ppg_signal)
            abp_denoised = wavelet.denoise(abp_signal)
            ecg_denoised = wavelet.denoise(ecg_signal)

            num_segments = ppg_denoised.shape[0] // segment_size
            total_segments += num_segments

            ppg_segments = []
            abp_segments = []
            ecg_segments = []
            segment_features = []

            for i in range(num_segments):
                start = i * segment_size
                end = start + segment_size

                if end <= ppg_denoised.shape[0] and not np.any(np.isnan(abp_denoised[start:end])) and not np.any(
                        np.isnan(ppg_denoised[start:end])) and not np.any(np.isnan(ecg_denoised[start:end])):

                    # Segmentation
                    ppg_segment = ppg_denoised[start:end]
                    abp_segment = abp_denoised[start:end]
                    ecg_segment = ecg_denoised[start:end]

                    # Filtering
                    ppg_filtered = lowpass_filter(ppg_segment, 60, 125, order=4)
                    abp_filtered = lowpass_filter(abp_segment, 60, 125, order=4)
                    ecg_filtered = lowpass_filter(ecg_segment, 60, 125, order=4)

                    # Normalization
                    ppg_norm = (ppg_filtered - np.mean(ppg_filtered)) / np.std(ppg_filtered)
                    ecg_norm = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)

                    if is_record_good(ppg_norm, 'ppg') and is_record_good(abp_filtered, 'abp') and is_record_good(
                            ecg_norm, 'ecg'):
                        ppg_segments.append(ppg_norm)
                        abp_segments.append(abp_filtered)
                        ecg_segments.append(ecg_norm)

                        f = extract_features(ppg_norm, ecg_norm, 125)
                        segment_features.append(f)

                else:
                    print("Failed the first if condition.")
                    continue

                final_segments += len(ppg_segment)

                ppg = np.vstack((ppg, ppg_segments))
                abp = np.vstack((abp, abp_segments))
                ecg = np.vstack((ecg, ecg_segments))
                features = np.vstack((features, segment_features))
            print(ppg.shape)
            print("segmented features shape: ", len(segment_features), len(segment_features[0]))

ppg = ppg[1:, :]
abp = abp[1:, :]
ecg = ecg[1:, :]
features = features[1:, :]

print(ppg.shape, abp.shape, ecg.shape, features.shape)
# print(ppg[0, :])
x = 0
for i in range(ppg.shape[0]):
    if np.any(np.isnan(features[i, :])):
        # print(features[i, :])
        x = x + 1
print(f"Total samples with nan values: {x}")

np.save("../preprocessed_dataset/ppg_uci.npy", ppg)
np.save("../preprocessed_dataset/abp_uci.npy", abp)
np.save("../preprocessed_dataset/ecg_uci.npy", ecg)
np.save("../preprocessed_dataset/features_uci.npy", features)

