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


class DKTDataSet(Dataset):
    def __init__(self, ques, delta, base, seen, repeat, pos, ans, mask):
        self.ques = ques
        self.delta = delta
        self.base = base
        self.seen = seen
        self.repeat = repeat
        self.pos = pos
        self.ans = ans
        self.mask = mask

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        deltas = self.delta[index]
        bases = self.base[index]
        seens = self.seen[index]
        repeats = self.repeat[index]
        poss = self.pos[index]
        answers = self.ans[index]
        mask = self.mask[index]
        questions = questions.unsqueeze(1)
        deltas = deltas.unsqueeze(1)
        bases = bases.unsqueeze(1)
        seens = seens.unsqueeze(1)
        repeats = repeats.unsqueeze(1)
        poss = poss.unsqueeze(1)
        answers = answers.unsqueeze(1)
        mask = mask.unsqueeze(1)
        combined_feature = torch.cat((questions, deltas, bases, seens, repeats, poss), dim=1)
        return combined_feature, answers, mask


class MomoDataSet(Dataset):
    def __init__(self, ques, delta, base, repeat, pos, ans, mask):
        self.ques = ques
        self.delta = delta
        self.base = base
        self.repeat = repeat
        self.pos = pos
        self.ans = ans
        self.mask = mask

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        deltas = self.delta[index]
        bases = self.base[index]
        repeats = self.repeat[index]
        poss = self.pos[index]
        answers = self.ans[index]
        mask = self.mask[index]
        questions = questions.unsqueeze(1)
        deltas = deltas.unsqueeze(1)
        bases = bases.unsqueeze(1)
        repeats = repeats.unsqueeze(1)
        poss = poss.unsqueeze(1)
        answers = answers.unsqueeze(1)
        mask = mask.unsqueeze(1)
        combined_feature = torch.cat((questions, deltas, bases, repeats, poss), dim=1)
        return combined_feature, answers, mask


class DataReader():
    def __init__(self, path, maxstep):
        self.path = path
        self.maxstep = maxstep

    def getdata(self):
        data_qus = np.array([])
        data_delta = np.array([])  # 时间戳
        data_base = np.array([])  # 记忆该单词的基础  history_correct / history_seen
        data_seen = np.array([])  # 记忆该单词见到的次数  session_seen
        data_repeat = np.array([])  # 记忆该单词时，已经见过多少次该单词， 这里指的是该数据集中出现的次数，不包含history_seen
        data_pos = np.array([])  # 记忆该单词时，上次见到该单词的时间间隔
        data_ans = np.array([])
        data_mask = np.array([])  # 填充标记

        with open(self.path, 'r') as f:

            if 'momo' in self.path:
                for len, ques, delta, base, repeat, pos, ans in tqdm.tqdm(itertools.zip_longest(*[f] * 7),
                                                                          desc=f'Processing ' + self.path[-17:-8],
                                                                          mininterval=2):
                    len = int(len.strip().strip(','))
                    ques = np.array(ques.strip().strip(',').split(','))
                    delta = np.array(delta.strip().strip(',').split(','))
                    base = np.array(base.strip().strip(',').split(','))
                    repeat = np.array(repeat.strip().strip(',').split(','))
                    pos = np.array(pos.strip().strip(',').split(','))
                    ans = np.array(ans.strip().strip(',').split(','))
                    mask = np.ones(len)
                    # print(ans)

                    mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                    zero = np.zeros(mod)
                    ques = np.append(ques, zero)
                    delta = np.append(delta, zero)
                    base = np.append(base, zero)
                    repeat = np.append(repeat, zero)
                    pos = np.append(pos, zero)
                    ans = np.append(ans, zero)
                    mask = np.append(mask, zero)
                    data_qus = np.append(data_qus, ques)
                    data_delta = np.append(data_delta, delta)
                    data_base = np.append(data_base, base)
                    data_repeat = np.append(data_repeat, repeat)
                    data_pos = np.append(data_pos, pos)
                    data_ans = np.append(data_ans, ans)
                    data_mask = np.append(data_mask, mask)

                data_qus = data_qus.astype(np.float32)
                data_delta = data_delta.astype(np.float32)
                data_base = data_base.astype(np.float32)
                data_repeat = data_repeat.astype(np.float32)
                data_pos = data_pos.astype(np.float32)
                data_ans = data_ans.astype(np.float32)
                data_mask = data_mask.astype(np.bool_)
                # data_ans = np.clip(data_ans, 0, np.inf)
                data_qus = torch.tensor(data_qus.reshape([-1, self.maxstep]))
                data_delta = torch.tensor(data_delta.reshape([-1, self.maxstep]))
                data_base = torch.tensor(data_base.reshape([-1, self.maxstep]))
                data_repeat = torch.tensor(data_repeat.reshape([-1, self.maxstep]))
                data_pos = torch.tensor(data_pos.reshape([-1, self.maxstep]))
                data_ans = torch.tensor(data_ans.reshape([-1, self.maxstep]))
                data_mask = torch.tensor(data_mask.reshape([-1, self.maxstep]))

                dateSet = MomoDataSet(data_qus, data_delta, data_base, data_repeat, data_pos, data_ans, data_mask)
                print('Finish processing ' + self.path[-11:-4])

            else:
                for len, ques, delta, base, seen, repeat, pos, ans in tqdm.tqdm(itertools.zip_longest(*[f] * 8),
                                                                                desc=f'Processing ' + self.path[-17:-8],
                                                                                mininterval=2):
                    len = int(len.strip().strip(','))
                    # if len > C.MAX_STEP:
                    ques = np.array(ques.strip().strip(',').split(','))
                    delta = np.array(delta.strip().strip(',').split(','))
                    base = np.array(base.strip().strip(',').split(','))
                    seen = np.array(seen.strip().strip(',').split(','))
                    repeat = np.array(repeat.strip().strip(',').split(','))
                    pos = np.array(pos.strip().strip(',').split(','))
                    ans = np.array(ans.strip().strip(',').split(','))
                    mask = np.ones(len)
                    # print(ans)

                    mod = 0 if len % self.maxstep == 0 else (self.maxstep - len % self.maxstep)
                    zero = np.zeros(mod)
                    ques = np.append(ques, zero)
                    delta = np.append(delta, zero)
                    base = np.append(base, zero)
                    seen = np.append(seen, zero)
                    repeat = np.append(repeat, zero)
                    pos = np.append(pos, zero)
                    ans = np.append(ans, zero)
                    mask = np.append(mask, zero)
                    data_qus = np.append(data_qus, ques)
                    data_delta = np.append(data_delta, delta)
                    data_base = np.append(data_base, base)
                    data_seen = np.append(data_seen, seen)
                    data_repeat = np.append(data_repeat, repeat)
                    data_pos = np.append(data_pos, pos)
                    data_ans = np.append(data_ans, ans)
                    data_mask = np.append(data_mask, mask)

                data_qus = data_qus.astype(np.float32)
                data_delta = data_delta.astype(np.float32)
                data_base = data_base.astype(np.float32)
                data_seen = data_seen.astype(np.float32)
                data_repeat = data_repeat.astype(np.float32)
                data_pos = data_pos.astype(np.float32)
                data_ans = data_ans.astype(np.float32)
                data_mask = data_mask.astype(np.bool_)
                # data_ans = np.clip(data_ans, 0, np.inf)
                data_qus = torch.tensor(data_qus.reshape([-1, self.maxstep]))
                data_delta = torch.tensor(data_delta.reshape([-1, self.maxstep])) / 1440
                data_base = torch.tensor(data_base.reshape([-1, self.maxstep]))
                data_seen = torch.tensor(data_seen.reshape([-1, self.maxstep]))
                data_repeat = torch.tensor(data_repeat.reshape([-1, self.maxstep]))
                data_pos = torch.tensor(data_pos.reshape([-1, self.maxstep])) / 1440
                data_ans = torch.tensor(data_ans.reshape([-1, self.maxstep]))
                data_mask = torch.tensor(data_mask.reshape([-1, self.maxstep]))

                dateSet = DKTDataSet(data_qus, data_delta, data_base, data_seen, data_repeat, data_pos, data_ans,
                                     data_mask)
                print('Finish processing ' + self.path[-11:-4])

        return dateSet


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
            if C.Torch_model_name == 'FIFKT':
                self.inputs: np.ndarray = datas['x']
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


if __name__ == "__main__":
    getLoader()

    # file = ['../dataset/duolinguo/it_data_max.txt',
    #         '../dataset/duolinguo/en_data_max.txt',
    #         '../dataset/duolinguo/pt_data_max.txt',
    #         '../dataset/duolinguo/es_data_max.txt',
    #         '../dataset/duolinguo/duolinguo_max.txt']
    #
    # files = ['../dataset/duolinguo/it_data_10/train.npy', '../dataset/duolinguo/it_data_10/val.npy',
    #          '../dataset/duolinguo/it_data_10/test.npy',
    #          '../dataset/duolinguo/en_data_10/train.npy', '../dataset/duolinguo/en_data_10/val.npy',
    #          '../dataset/duolinguo/en_data_10/test.npy',
    #          '../dataset/duolinguo/pt_data_10/train.npy', '../dataset/duolinguo/pt_data_10/val.npy',
    #          '../dataset/duolinguo/pt_data_10/test.npy',
    #          '../dataset/duolinguo/es_data_10/train.npy', '../dataset/duolinguo/es_data_10/val.npy',
    #          '../dataset/duolinguo/es_data_10/test.npy',
    #          '../dataset/duolinguo/duolinguo_data_10/train.npy', '../dataset/duolinguo/duolinguo_data_10/val.npy',
    #          '../dataset/duolinguo/duolinguo_data_10/test.npy']

    # file = ['../dataset/momo/momo_data_forget.txt']
    # files = ['../dataset/momo/momo_data_forget/train.npy', '../dataset/momo/momo_data_forget/val.npy',
    #          '../dataset/momo/momo_data_forget/test.npy']
    #
    # amaxstep = C.MAX_STEP
    # c = 0
    #
    # for f in file:
    #     handel = DataReader(f, amaxstep)
    #     dataset = handel.getdata()
    #     train_loader, val_loader, test_loader = data_loader(dataset)
    #     print('Start save ' + f[-15:-8] + ' train, val, test dataset')
    #     torch.save(train_loader, files[c * 3])
    #     torch.save(val_loader, files[c * 3 + 1])
    #     torch.save(test_loader, files[c * 3 + 2])
    #
    #     for batch, (data, label, mask) in enumerate(train_loader):
    #         print(batch)
    #         print(data.shape)
    #         print(label.shape)
    #         print(mask.shape)
    #     print('Finish save ' + f[-11:-4] + ' train, val, test dataset')
    #     c += 1
