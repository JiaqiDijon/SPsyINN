import torch
from torch.utils.data import DataLoader
import model.Constants as C


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, z, m):
        self.X = X
        self.y = y
        self.z = z
        self.m = m

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.z[index], self.m[index]

    def __len__(self):
        return len(self.X)


class NewDataste(torch.utils.data.Dataset):
    def __init__(self, X, y, z):
        self.X = X
        self.y = y
        self.z = z

    # z The prediction labels for DKT or SR, used to influence each other.
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.z[index]

    def __len__(self):
        return len(self.X)


def save_DKT_pred(X, y, z):
    SampleLoaders = DataLoader(NewDataste(X, y, z), batch_size=C.BATCH_SIZE, shuffle=True)
    torch.save(SampleLoaders, C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'NN_pred.npy')


def save_PYSR_pred(X, y, z):
    SampleLoaders = DataLoader(NewDataste(X, y, z), batch_size=C.BATCH_SIZE, shuffle=True)
    torch.save(SampleLoaders, C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GPSR_pred.npy')
