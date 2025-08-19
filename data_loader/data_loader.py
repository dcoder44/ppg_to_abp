from data_loader.dataset import BPDataset
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
from torch.utils.data import Subset


def data_loader(batch_size=32, shuffle=True, val_split=0.9, generator=None, mode="train", max_samples=None):
    print(f"Loading the {mode} dataset...")
    dataset = BPDataset(mode=mode, max_samples=max_samples)

    if val_split < 1.0:
        train_size = int(val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_dataset = dataset
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = None
        val_dataset = None
    
    print(f"Loaded the dataset successfully. Total samples: {len(dataset)}")
    return train_dataset, val_dataset, train_loader, val_loader


def get_nfold_indices(n_splits=5, mode="n-fold", max_samples=None):
    dataset = BPDataset(mode=mode, max_samples=max_samples)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return kf.split(range(len(dataset)))

def get_nfold_data_loaders(n_splits=5, batch_size=64, val_split=0.9, shuffle=True, mode="n-fold", max_samples=None, generator=None):
    print(f"Loading the dataset for {n_splits}-fold cross-validation...")
    indices = get_nfold_indices(n_splits=n_splits, mode=mode, max_samples=max_samples)
    fold_loaders = []
    
    dataset = BPDataset(mode=mode, max_samples=max_samples)

    for train_indices, test_indices in indices:
        
        test_dataset = Subset(dataset, test_indices)
        
        t_dataset = Subset(dataset, train_indices)
        train_size = int(val_split * len(t_dataset))
        val_size = len(t_dataset) - train_size
        train_dataset, val_dataset = random_split(t_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)

        # test_dataset = BPDataset(mode=mode, max_samples=max_samples)
        # test_dataset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

        fold_loaders.append((train_loader, val_loader, test_loader))

    print(f"Loaded the dataset successfully. Total samples: {len(dataset)}")
    return fold_loaders
