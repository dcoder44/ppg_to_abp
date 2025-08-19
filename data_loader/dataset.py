from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class BPDataset(Dataset):
    def __init__(self, mode="train", max_samples=None):
        self.abp = np.zeros([1, 512])
        self.ppg = np.zeros([1, 512])
        self.features = np.zeros([1, 26])

        self.mode = mode
        self.max_samples = max_samples

        self.load_and_preprocess_data()

        self.num_samples = self.abp.shape[0]

    def load_and_preprocess_data(self, waveform_dir="preprocessed_dataset"):
        if self.mode == "train":
            self.abp = np.load(f"{waveform_dir}/abp_train.npy")
            self.ppg = np.load(f"{waveform_dir}/ppg_train.npy")
            self.features = np.load(f"{waveform_dir}/features_train.npy")
        
        elif self.mode == "test":
            self.abp = np.load(f"{waveform_dir}/abp_test.npy")
            self.ppg = np.load(f"{waveform_dir}/ppg_test.npy")
            self.features = np.load(f"{waveform_dir}/features_test.npy")

        if self.mode == "mimic4_train":
            print("Loading the mimic4 training dataset...")
            self.abp = np.load(f"mimic4_preprocessed/abp_mimic4_train.npy")
            self.ppg = np.load(f"mimic4_preprocessed/ppg_mimic4_train.npy")
            self.features = np.load(f"mimic4_preprocessed/features_mimic4_train.npy")
        
        elif self.mode == "mimic4_test":
            print("Loading the mimic4 test dataset...")
            self.abp = np.load(f"mimic4_preprocessed/abp_mimic4_test.npy")
            self.ppg = np.load(f"mimic4_preprocessed/ppg_mimic4_test.npy")
            self.features = np.load(f"mimic4_preprocessed/features_mimic4_test.npy")

        elif self.mode == "n-fold":
            abp_test = np.load(f"{waveform_dir}/abp_test.npy")
            ppg_test = np.load(f"{waveform_dir}/ppg_test.npy")
            features_test = np.load(f"{waveform_dir}/features_test.npy")
            abp_train = np.load(f"{waveform_dir}/abp_train.npy")
            ppg_train = np.load(f"{waveform_dir}/ppg_train.npy")
            features_train = np.load(f"{waveform_dir}/features_train.npy")
            self.abp = np.concatenate((abp_train, abp_test), axis=0)
            self.ppg = np.concatenate((ppg_train, ppg_test), axis=0)
            self.features = np.concatenate((features_train, features_test), axis=0)

        if self.max_samples is not None:
            self.abp = self.abp[:self.max_samples, :]
            self.ppg = self.ppg[:self.max_samples, :]
            self.features = self.features[:self.max_samples, :]

        self.features[:, 5] = np.log1p(self.features[:, 5])
        self.features[:, 12] = np.log1p(self.features[:, 12])
        self.features[:, 18] = np.log1p(self.features[:, 18])

        scaler = StandardScaler()
        # Right now fit_transform is for every new feature set. We should fit while training and use it while inferencing...
        self.features = scaler.fit_transform(self.features)

        self.abp = torch.from_numpy(self.abp)
        self.ppg = torch.from_numpy(self.ppg)
        self.features = torch.from_numpy(self.features)
        print("Loaded the dataset.")

    def __getitem__(self, index):
        return self.abp[index, :], self.ppg[index, :], self.features[index, :]

    def __len__(self):
        return self.num_samples
