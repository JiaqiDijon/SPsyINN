import os
import tqdm
import time
import torch
import pickle
import numpy as np
import pandas as pd
import model.Constants as C
from pysr.sr import PySRRegressor
from model.Sample_data import save_PYSR_pred
from model.data_loader import getLoader, DKTDataSet, MomoDataSet
import warnings
warnings.filterwarnings("ignore")

equation_file = np.array([
    "x0 + log(pow(x1, -2 * x5))",
    "exp((-2 * x2) / (x2 + x5))",
    "x3 * pow((1 + x1 * x2), -1 * x0)",
    "pow(2,(-4 * x1 / pow(2, x2 / x4)))",
    "x3 * pow((2 + x1 * 2), -1 * x0) * exp(-1 * x1)",
])
'''
# ACT-R 
# Wozniak 
# Wixted 
# HLR  
# Wickelgren
'''

def masked_mae(preds, labels, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if torch.isnan(torch.tensor(null_val)):
            mask = ~torch.isnan(labels)
        else:
            mask = labels != null_val

        mask = torch.tensor(mask, dtype=torch.float32)
        mask /= torch.mean(mask)

        mae = torch.abs(preds - labels)
        mae = torch.nan_to_num(mask * mae)
        return torch.mean(mae)


def masked_mape(preds, labels, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if torch.isnan(torch.tensor(null_val)):
            mask = ~torch.isnan(labels)
        else:
            mask = labels != null_val

        mask = torch.tensor(mask, dtype=torch.float32)
        mask /= torch.mean(mask)

        # Calculate MAPE
        mape = torch.abs((preds - labels) / labels)
        mape = torch.nan_to_num(mask * mape)

        return torch.mean(mape) * 100


def masked_rmse(preds, labels, null_val=0):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=0))


def masked_mse(preds, labels, null_val=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        if torch.isnan(torch.tensor(null_val)):
            mask = ~torch.isnan(labels)
        else:
            mask = labels != null_val

        mask = torch.tensor(mask, dtype=torch.float32)
        mask /= torch.mean(mask)

        mse = torch.square(preds - labels)
        mse = torch.nan_to_num(mse * mask)
        return torch.mean(mse)


def performance(ground_truth, prediction):
    # r2 = torch.tensor(r2_score(ground_truth.detach().numpy(), prediction.detach().numpy()))
    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    mse = masked_mse(prediction, ground_truth)
    rmse = masked_rmse(prediction, ground_truth)
    result = f'Py-SR:   MAE: {mae.item()} MAPE: {mape.item()} MSE: {mse.item()} RMSE: {rmse.item()}\n'
    result += '----------------------------------------------------------------------------------\n'
    with open(C.Dpath + C.DATASET + '/PySRresult.txt', 'a+', encoding='utf-8') as f:
        f.write(result)
    print(result)


def save_epoch(epoch, ground_truth, prediction):
    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    mse = masked_mse(prediction, ground_truth)
    rmse = masked_rmse(prediction, ground_truth)
    result = f'{epoch} :   MAE: {mae.item()} MAPE: {mape.item()} MSE: {mse.item()} RMSE: {rmse.item()}\n'
    with open(C.Dpath + C.DATASET + '/GPSR-epoch.txt', 'a+', encoding='utf-8') as f:
        f.write(result)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path=C.Dpath + C.DATASET + '/Pysr_checkpoint.pkl'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        with open(self.save_path, 'wb') as f:
            pickle.dump(model, f)
        self.val_loss_min = val_loss

    def load_checkpoint(self):
        with open(self.save_path, 'rb') as f:
            model = pickle.load(f)
        return model


def train(model, trainLoaders, early_stopping, testLoaders):
    X = torch.Tensor([])
    y = torch.Tensor([])
    Data = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(trainLoaders):
        X = torch.cat([X, data[:, -1, :]], dim=0)
        y = torch.cat([y, label], dim=0)
        Data = torch.cat([Data, data], dim=0)

    num_samples = 1024
    indices = torch.randperm(X.size(0))[:num_samples]  # 生成一个随机排列并取前1024个索引

    min = torch.min(Data.reshape(-1, C.INPUT), dim=0)[0]
    max = torch.max(Data.reshape(-1, C.INPUT), dim=0)[0]

    X = (X - min) / (max - min)

    X = X[indices]  # 使用随机索引抽取样本
    y_sr = y[indices]
    X = torch.nan_to_num(X, 0)

    if C.Torch_model_name == 'FIFKT':
        if 'momo' not in C.DATASET:
            model.fit(X[:, [3, 5, 6, 8, 9, 10]].detach().numpy(), y_sr.detach().numpy())
        else:
            model.fit(X.detach().numpy(), y_sr.detach().numpy())
    else:
        model.fit(X.detach().numpy(), y_sr.detach().numpy())
    test(model, testLoaders, min, max)

    model, val_loss = val(model, valLoaders, min, max)
    # 调用早停机制
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
    model = early_stopping.load_checkpoint()

    indices_forDNN = torch.randperm(Data.size(0))[:num_samples]
    Data = Data[indices_forDNN]  # 使用随机索引抽取样本
    y = y[indices_forDNN]
    Data = (Data - min) / (max - min)

    Data = torch.nan_to_num(Data, 0)

    if C.Torch_model_name == 'FIFKT':
        if 'momo' not in C.DATASET:
            pred = torch.Tensor([model.predict(Data[:, -1, [3, 5, 6, 8, 9, 10]].cpu().detach().numpy())]).squeeze(
                0).unsqueeze(1)
        else:
            pred = torch.Tensor([model.predict(Data[:, -4, :].cpu().detach().numpy())]).squeeze(0).unsqueeze(1)
    else:
        pred = model.predict(Data[:, -1, :])

    save_PYSR_pred(Data, y, pred.reshape(pred.shape[0], 1))
    if os.path.exists(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'NN_pred.npy'):
        D = torch.Tensor([])
        L = torch.Tensor([])
        DKTpred = torch.load(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'NN_pred.npy')
        for batch_idx, (data, label, dktpred) in enumerate(DKTpred):
            data = torch.nan_to_num(data, 0)
            if C.Torch_model_name == 'FIFKT':
                if 'momo' not in C.DATASET:
                    D = torch.cat([D, data[:, -1, [3, 5, 6, 8, 9, 10]].cpu()], dim=0)
                    L = torch.cat([L, dktpred.cpu()], dim=0)
                else:
                    D = torch.cat([D, data[:, -1, :].cpu()], dim=0)
                    L = torch.cat([L, dktpred.cpu()], dim=0)
            else:
                D = torch.cat([D, data[:, -1, :].cpu()], dim=0)
                L = torch.cat([L, dktpred[:, -1, :].cpu()], dim=0)

        model.fit(D.cpu().detach().numpy(), L.cpu().detach().numpy())

    return model, min, max


def val(model, valLoaders, min, max):
    X = torch.Tensor([])
    y = torch.Tensor([])
    prediction = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(valLoaders):
        X = torch.cat([X, data], dim=0)
        y = torch.cat([y, label], dim=0)
        data = (data - min) / (max - min)
        data = data[:, -1, :]
        data = torch.nan_to_num(data, 0)
        if C.Torch_model_name == 'FIFKT':
            if 'momo' not in C.DATASET:
                pred = torch.Tensor([model.predict(data[:, [3, 5, 6, 8, 9, 10]].cpu().detach().numpy())]).squeeze(
                    0).unsqueeze(1)
            else:
                pred = torch.Tensor([model.predict(data.cpu().detach().numpy())]).squeeze(0).unsqueeze(1)
        else:
            pred = torch.Tensor([model.predict(data.cpu().detach().numpy())]).squeeze(0).unsqueeze(
                1)  # .reshape(prediction.shape[1], prediction.shape[0])
        prediction = torch.cat([prediction, pred], dim=0)

    # 计算验证损失
    val_loss = torch.nn.functional.mse_loss(prediction, y)
    return model, val_loss.item()


def test(model, testLoaders, min, max, last=False):
    ground_truth = torch.Tensor([])
    prediction = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(testLoaders):
        data = data[:, -1, :]
        data = (data - min) / (max - min)
        ground_truth = torch.cat([ground_truth, label.cpu()])
        data = torch.nan_to_num(data, 0)
        if C.Torch_model_name == 'FIFKT':
            if 'momo' not in C.DATASET:
                SR_pred = torch.Tensor([model.predict(data[:, [3, 5, 6, 8, 9, 10]].cpu().detach().numpy())])
            else:
                SR_pred = torch.Tensor([model.predict(data.cpu().detach().numpy())])
        else:
            SR_pred = torch.Tensor([model.predict(data.cpu().detach().numpy())])
        prediction = torch.cat([prediction, SR_pred.squeeze(0).unsqueeze(1)], dim=0)

    print(prediction.max())
    print(prediction.min())
    print(ground_truth.max())
    print(ground_truth.min())
    performance(ground_truth, prediction)
    if last == True:
        return ground_truth, prediction

#
# if __name__ == '__main__':
#
#     # 尝试读取 ALL.csv 文件
#     try:
#         e_df = pd.read_csv(C.functionlist[C.Function])
#     except pd.errors.ParserError as e:
#         print(f"Error reading ALL.csv: {e}")
#         raise
#
#     # 确保读取成功后，将数据写入 equcation.csv 文件，替换原有内容
#     e_df.to_csv('equcation.csv', index=False)
#
#     print(f"{C.functionlist[C.Function]}的数据已成功写入 equcation.csv 文件中，原有内容已被替换。")
#
#     print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')
#     model = PySRRegressor(
#         niterations=40,
#         binary_operators=["+", "*", "-", "/", "pow"],
#         populations=40,
#         population_size=50,
#         ncycles_per_iteration=500,
#         equation_file='equcation.csv',
#         batching=True,
#         # select_k_features=4,
#         batch_size=256,
#         precision=64,
#         loss="L2DistLoss()",
#         maxsize=10,
#         warm_start=True,
#         maxdepth=4,  # 避免深度嵌套
#         procs=20,
#         annealing=True,
#         alpha=0.1,
#         unary_operators=[
#             "exp",
#             "log"],
#         # unary_operators=[
#         #     # "exp",
#         #     "exp"],
#         model_selection="accuracy",  # score、accuracy、best
#         early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
#     )
#     trainLoaders, valLoaders, testLoaders = getLoader()
#
#     # 初始化早停机制
#     early_stopping = EarlyStopping(patience=5, verbose=True)
#
#     for epoch in tqdm.tqdm(range(C.EPOCH), 'training...'):
#         start_time = time.time()
#         model, min, max = train(model, trainLoaders, early_stopping, testLoaders)
#         print(model)
#         print(f'epoch :{epoch}, loaded best model')
#         ground_truth, prediction = test(model, testLoaders, min, max, last=True)
#         save_epoch(epoch, ground_truth, prediction)
#         model.equations.to_csv(C.Dpath + C.DATASET + '/' + C.Function + str(epoch) + 'function.csv')
#         end_time = time.time()
#         epoch_time = end_time - start_time
#         with open(C.Dpath + C.DATASET + '/GPSR-time.txt', 'w+', encoding='utf-8') as f:
#             f.write(str(epoch_time))
#         if os.path.exists(C.Dpath + C.DATASET + '/' + C.Torch_model_name + '-time.txt'):
#             with open(C.Dpath + C.DATASET + '/' + C.Torch_model_name + '-time.txt', 'r', encoding='utf-8') as f:
#                 DDKTtime = f.read()
#                 if float(DDKTtime) > epoch_time:
#                     time.sleep(float(DDKTtime) - epoch_time)
#         else:
#             time.sleep(0)
#
#     # 训练结束后，加载最佳模型
#     model = early_stopping.load_checkpoint()
#     print("Loaded best model")
#     ground_truth, prediction = test(model, testLoaders, min, max, last=True)
#     model.equations.to_csv(C.Dpath + C.DATASET + '/' + C.Function + 'PySRfunction.csv')
