import torch
from torch.utils.data.dataset import Dataset
path = 'Data/'


class MyDataste(Dataset):
    def __init__(self, Data):
        self.x = Data[:, :, [0, 1, 2, 3, 4, 5, 6, 7]]
        self.y = Data[:, :, 8]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def normalize_data(data, min_val, max_val):
    # 避免除零错误
    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-7  # 如果范围为0，防止除零
    normalized_data = (data - min_val) / range_val
    return normalized_data

class CustomNormalizedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, original_loader, min_val, max_val, column_order=None):
        self.original_loader = original_loader
        self.min_val = min_val
        self.max_val = max_val
        self.column_order = column_order

    def __iter__(self):
        for data, label in self.original_loader:
            # 数据归一化
            normalized_data = normalize_data(data, self.min_val, self.max_val)

            # 如果指定了列的顺序或选择，则进行调整
            if self.column_order is not None:
                normalized_data = normalized_data[:, :, self.column_order]

            yield normalized_data, label

    def __len__(self):
        return len(self.original_loader.dataset)


def get_min_max(trainLoaders):
    # 初始化最大最小值
    min_val, max_val = None, None

    for batch_idx, (data, label) in enumerate(trainLoaders):
        # 仅考虑需要的维度
        data_flat = data.reshape(-1, data.shape[2])

        # 计算当前批次最大值和最小值
        batch_min = torch.min(data_flat, dim=0)[0]
        batch_max = torch.max(data_flat, dim=0)[0]

        # 更新全局最大最小值
        min_val = batch_min if min_val is None else torch.min(min_val, batch_min)
        max_val = batch_max if max_val is None else torch.max(max_val, batch_max)

    return max_val, min_val

def getLoaders(Dataname):
    trainLoader = torch.load(path + Dataname + '/train.npz')
    testLoader = torch.load(path + Dataname + '/test.npz')
    valLoader = torch.load(path + Dataname + '/val.npz')
    return trainLoader, testLoader, valLoader
