import numpy as np
import itertools
import tqdm
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import model.Constants as C

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2


def data_loader(dateSet):
    train_ratio = 0.7
    test_ratio = 0.2
    size = len(dateSet)
    train_size = int(size * train_ratio)
    test_size = int(size * test_ratio)
    val_size = size - train_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dateSet, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=C.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=C.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=C.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


class PredictorDataset(Dataset):
    def __init__(self, data_path: str, ):
        datas = np.load(data_path)

        if 'momo' in C.DATASET:
            self.inputs: np.ndarray = datas['x']
            self.targets: np.ndarray = datas['y']
        else:
            self.inputs: np.ndarray = datas['x'][:, :, [3, 5, 6, 8, 9, 10]]
            self.targets: np.ndarray = datas['y']
            self.x: np.ndarray = datas
        # min = np.min(self.inputs.reshape(-1, 11), axis=0)
        # max = np.max(self.inputs.reshape(-1, 11), axis=0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x, y = self.inputs[idx], self.targets[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __minmax__(self):
        min = np.min(self.inputs.reshape(-1, 11), axis=0)
        max = np.max(self.inputs.reshape(-1, 11), axis=0)
        return torch.tensor(min, dtype=torch.float32), torch.tensor(max, dtype=torch.float32)


def getLoader():
    traindataset = PredictorDataset(C.Dpath + C.DATASET + '/train.npz')
    testdaraset = PredictorDataset(C.Dpath + C.DATASET + '/test.npz')
    valdataset = PredictorDataset(C.Dpath + C.DATASET + '/val.npz')

    trainLoaders = DataLoader(traindataset, batch_size=C.BATCH_SIZE, shuffle=True)
    testLoaders = DataLoader(testdaraset, batch_size=C.BATCH_SIZE, shuffle=False)
    valLoaders = DataLoader(valdataset, batch_size=C.BATCH_SIZE, shuffle=False)

    # trainLoaders = torch.load(C.Dpath + C.DATASET + '/train.npy')
    # testLoaders = torch.load(C.Dpath + C.DATASET + '/test.npy')
    # valLoaders = torch.load(C.Dpath + C.DATASET + '/val.npy')

    return trainLoaders, testLoaders, valLoaders

