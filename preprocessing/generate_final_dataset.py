import numpy as np

print("Loading the files...")

ppg1 = np.load("../preprocessed_dataset/features_uci_1.npy")
ppg2 = np.load("../preprocessed_dataset/features_uci_2.npy")
ppg4 = np.load("../preprocessed_dataset/features_uci_4.npy")

print("Files loaded. Stacking the files...")

ppg1 = np.vstack((ppg1, ppg2))
ppg1 = np.vstack((ppg1, ppg4))

print("Files stacked. Saving the final file...")

np.save("../preprocessed_dataset/features_uci_124.npy", ppg1)

print("Done!")
