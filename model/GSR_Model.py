import os
import torch
import pickle
import numpy as np
import model.Constants as C
from model.Sample_data import save_PYSR_pred
import warnings
warnings.filterwarnings("ignore")

def compute_rmse_in_bins(predictions, targets, num_bins=10):
    """
    根据分区间计算加权 RMSE (Root Mean Square Error in Bins)

    参数:
    - predictions: 模型的预测值 (torch.Tensor)
    - targets: 真实标签值 (torch.Tensor)
    - num_bins: 要划分的区间数 (int)

    返回:
    - 计算的 RMSE 值 (torch.Tensor)
    """
    # 确保预测值和真实值具有相同形状
    assert predictions.shape == targets.shape, "预测值和真实值的形状必须一致"

    # 获取预测值的最小值和最大值，作为区间划分的依据
    min_value = torch.min(predictions).item()
    max_value = torch.max(predictions).item()

    # 生成区间的边界
    bins = torch.linspace(min_value, max_value, num_bins + 1)

    weighted_sum = 0  # 用于存储公式中的 ∑w_i * (avg_pred - avg_true)^2
    total_weight = 0  # 用于存储 ∑w_i

    # 遍历每个区间
    for i in range(num_bins):
        # 获取该区间的边界
        lower_bound = bins[i]
        upper_bound = bins[i + 1]

        # 筛选出落在该区间内的预测值和真实值
        mask = (predictions >= lower_bound) & (predictions < upper_bound)

        # 如果该区间没有数据，跳过该区间
        if torch.sum(mask) == 0:
            continue

        # 计算该区间内的权重 w_i（即该区间内的样本数）
        w_i = torch.sum(mask).item()

        # 计算该区间内预测值的平均值和真实值的平均值
        avg_pred = torch.mean(predictions[mask])
        avg_true = torch.mean(targets[mask])

        # 根据公式计算该区间的加权平方误差
        weighted_square_error = w_i * (avg_pred - avg_true) ** 2

        # 更新加权和与总权重
        weighted_sum += weighted_square_error
        total_weight += w_i

    # 计算加权 RMSE
    if total_weight == 0:  # 如果所有的区间都没有数据
        return torch.tensor(0.0)

    rmse = torch.sqrt(weighted_sum / total_weight)

    return rmse


def performance(ground_truth, prediction):

    prediction = prediction.squeeze(-1)

    mae = mean_absolute_error(ground_truth, prediction)
    rmse = root_mean_squared_error(ground_truth, prediction)
    # rmsebin = compute_rmse_in_bins(prediction, ground_truth)
    rmsebin = 0
    mse = mean_squared_error(ground_truth, prediction)
    r2 = r2_score(ground_truth, prediction)

    ground_truth[ground_truth < 1] = 0
    AUC = roc_auc_score(ground_truth, prediction)
    prediction[prediction < 0.5] = 0
    prediction[prediction >= 0.5] = 1
    recall = recall_score(ground_truth, prediction)
    precision = precision_score(ground_truth, prediction)
    F1 = f1_score(ground_truth, prediction)
    acc = accuracy_score(ground_truth, prediction)
    Precision, Recall, _ = precision_recall_curve(ground_truth, prediction)
    prauc = auc(Recall, Precision)
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction).ravel()
    Specificity = tn / (tn + fp)
    Sensitivity = tp / (tp + fn)
    G_Mean = np.sqrt(Sensitivity * Specificity)
    result = f'R2: {r2}, MAE: {mae}, RMSE: {rmse}, RMSE(bin): {rmsebin}, AUC: {AUC}, PrAUC: {prauc}, MSE: {mse}, ACC: {acc}, F1: {F1}, recall: {recall}, ' \
             f'precision: {precision} Specificity: {Specificity}, Sensitivity: {Sensitivity}, G_Mean: {G_Mean}\n'
    result += '----------------------------------------------------------------------------------\n'
    with open(C.Dpath + C.DATASET + '/GSR-epoch.txt', 'a+', encoding='utf-8') as f:
        f.write(result)
    print(result)

def save_epoch(epoch, ground_truth, prediction):
    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    result = f'{epoch} :   MAE: {mae.item()} MAPE: {mape.item()}\n'
    with open(C.Dpath + C.DATASET + '/GSR-epoch.txt', 'a+', encoding='utf-8') as f:
        f.write(result)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path=C.Dpath + C.DATASET + '/GSR_checkpoint.pkl'):
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

def get_XDATA(trainLoaders):
    X = torch.Tensor([])
    y = torch.Tensor([])
    Data = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(trainLoaders):
        if 'momo' in C.DATASET:
            X = torch.cat([X, data[:, -1, :-1]], dim=0)
            y = torch.cat([y, label], dim=0)
        else:
            X = torch.cat([X, data[:, -1, [3, 5, 6, 8, 9, 10]]], dim=0)
            y = torch.cat([y, label], dim=0)
        Data = torch.cat([Data, data], dim=0)
    return X, y, Data

def train(model, early_stop, X, y, Data):

    num_samples = 1024
    indices = torch.randperm(X.size(0))[:num_samples]  # 生成一个随机排列并取前1024个索引

    X = X[indices]  # 使用随机索引抽取样本
    y_sr = y[indices]
    X = torch.nan_to_num(X, 0)

    model.fit(X.detach().numpy(), y_sr.detach().numpy())

    model, val_loss = val(model, valLoaders)
    # 调用早停机制
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
    model = early_stopping.load_checkpoint()

    indices_forDNN = torch.randperm(Data.size(0))[:num_samples]
    Data = Data[indices_forDNN]  # 使用随机索引抽取样本

    if 'momo' in C.DATASET:
        pred = model.predict(Data[:, -1, :-1])
    else:
        pred = model.predict(Data[:, -1, [3, 5, 6, 8, 9, 10]])
    y = y[indices_forDNN]

    save_PYSR_pred(Data, y, pred.reshape(pred.shape[0], 1))
    if os.path.exists(C.Dpath + C.DATASET + '/' + C.modelname + 'NN_pred.npy'):
        D = torch.Tensor([])
        L = torch.Tensor([])
        DKTpred = torch.load(C.Dpath + C.DATASET + '/' + C.modelname + 'NN_pred.npy')
        for batch_idx, (data, label, dktpred) in enumerate(DKTpred):
            if 'momo' in C.DATASET:
                data = data[:, -1, :-1]
            else:
                data = data[:, -1, [3, 5, 6, 8, 9, 10]]

            D = torch.cat([D, data.cpu()], dim=0)
            L = torch.cat([L, dktpred[:, -1, :].cpu()], dim=0)

        model.fit(D.cpu().detach().numpy(), L.cpu().detach().numpy())

    return model


def val(model, valLoaders):
    y = torch.Tensor([])
    prediction = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(valLoaders):
        if 'MaiMemo' in C.DATASET:
            data = data[:, -1, :-1]
        else:
            data = data[:, -1, [3, 5, 6, 8, 9, 10]]
        pred = torch.Tensor([model.predict(data.cpu().detach().numpy())]).squeeze(0).unsqueeze(
            1)  # .reshape(prediction.shape[1], prediction.shape[0])
        y = torch.cat([y, label], dim=0)
        prediction = torch.cat([prediction, pred], dim=0)

    # 计算验证损失
    val_loss = torch.nn.functional.mse_loss(prediction, y)
    return model, val_loss.item()

def test(model, testLoaders, last=False):
    ground_truth = torch.Tensor([])
    prediction = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(testLoaders):
        if 'MaiMemo' in C.DATASET:
            data = data[:, -1, :-1]
        else:
            data = data[:, -1, [3, 5, 6, 8, 9, 10]]
        ground_truth = torch.cat([ground_truth, label.cpu()])
        SR_pred = torch.Tensor([model.predict(data.cpu().detach().numpy())])
        prediction = torch.cat([prediction, SR_pred.squeeze(0).unsqueeze(1)], dim=0)

    print(prediction.max())
    print(prediction.min())
    print(ground_truth.max())
    print(ground_truth.min())
    performance(ground_truth, prediction)
    if last == True:
        return ground_truth, prediction

