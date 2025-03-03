import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.Sample_data import save_DKT_pred
import model.Constants as C
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True
CUDA_LAUNCH_BLOCKING = 1

# noisy model
class DF_block(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_steps=100, beta_start=1e-3, beta_end=0.2):
        super(DF_block, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.noise_level = nn.Parameter(torch.tensor(0.5))

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alpha_hat[t].to(x.device)
        alpha_t_sqrt = torch.sqrt(alpha_t).view(-1, 1, 1)
        noise_t = noise * torch.sqrt(1 - alpha_t).view(-1, 1, 1)
        noise_t = noise_t * self.noise_level

        noisy_data = alpha_t_sqrt * x + noise_t
        return noisy_data.squeeze(0), noise

# DNN model
class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DNN, self).__init__()
        self.noise_module = DF_block(input_dim, hidden_dim, num_layers=5)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.q_emb = nn.Embedding(12326, 64)
        self.p_emb = nn.Embedding(20962, 64)
        self.a_emb = nn.Embedding(2, 64)

        # self.rnn = nn.LSTM(input_dim + 128, hidden_dim, layer_dim, batch_first=True, dropout=0.3)
        if 'momo' in C.DATASET:
            self.rnn = nn.LSTM(192 + 6, hidden_dim, layer_dim, batch_first=True, dropout=0.3)
        else:
            self.rnn = nn.LSTM(192 + 6, hidden_dim, layer_dim, batch_first=True, dropout=0.3)

        self.Linear = nn.Linear(64, 64)

        self.out = nn.Sequential(
            nn.Linear(64, 10),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(10, 1)
        )

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.t = 3

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)).to(x.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)).to(x.device)
        if 'momo' in C.DATASET:
            xnew = x
            a_data = x[:, :, 5].unsqueeze(-1)
            q_data = x[:, :, 0].unsqueeze(-1)
            p_data = x[:, :, 0].unsqueeze(-1)
        else:
            xnew = x[:, :, [3, 5, 6, 8, 9, 10]]
            q_data = x[:, :, 4].unsqueeze(-1)
            p_data = x[:, :, 0].unsqueeze(-1)
            a_data = x[:, :, 1].unsqueeze(-1)

        noisy_data, noise = self.noise_module(xnew, t=10)

        # word embeding
        q_embed_data = self.q_emb(q_data.to(dtype=torch.long)).squeeze()
        # user embeding
        p_embed_data = self.p_emb(p_data.to(dtype=torch.long)).squeeze()

        # ans embeding
        a_embed_data = self.a_emb(a_data.to(dtype=torch.long)).squeeze()


        # 对原始数据的预测
        original_data = torch.cat([q_embed_data, p_embed_data, a_embed_data, xnew], dim=2)
        original, (h0, c0) = self.rnn(original_data, (h0, c0))
        h = h0[-1:, :, :].permute(1, 0, 2)
        original = self.Linear(original + h)
        out_original = self.out(original)
        out_original = self.sig(out_original)

        # 对添加噪声数据的预测
        noisy_data = torch.cat([q_embed_data, p_embed_data, a_embed_data, noisy_data], dim=2)
        noisy, (h0, c0) = self.rnn(noisy_data, (h0, c0))
        h = h0[-1:, :, :].permute(1, 0, 2)
        noisy = self.Linear(noisy + h)
        out_noisy = self.out(noisy)
        out_noisy = self.sig(out_noisy)

        return out_original, out_noisy

# Loss Function
def Loss(y_true, y_pred_original, y_pred_noisy=None, y_sr=None, weight=False):
    maeloss = nn.L1Loss()
    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()
    if 'momo' not in C.DATASET:
        y_true = y_true.unsqueeze(1)
    if C.modelname in ['DKT']:
        y_pred_original = y_pred_original[:, -1, :]
        if y_sr is not None:
            loss_sr = mseloss(y_pred_original, y_sr)
            loss_original = mseloss(y_pred_original, y_true)
            if C.DyOp:
                orginal_true_loss = maeloss(y_pred_original, y_true)
                sr_true_loss = maeloss(y_sr, y_true)

                original_weight = sr_true_loss / (sr_true_loss + orginal_true_loss)
                sr_weight = orginal_true_loss / (sr_true_loss + orginal_true_loss)

                total_loss = original_weight * loss_original + sr_weight * loss_sr
            else:
                total_loss = 0.5 * loss_original + 0.5 * loss_sr
        else:
            total_loss = mseloss(y_pred_original, y_true)
        return total_loss

    elif C.modelname == 'DTKT':
        if C.Noisy:
            y_pred_original = y_pred_original[:, -1, :]
            y_pred_noisy = y_pred_noisy[:, -1, :]

            loss_original = bceloss(y_pred_original, y_true)
            loss_pred = mseloss(y_pred_original, y_pred_noisy)

            orginal_true_loss = mseloss(y_pred_original, y_true)
            noisy_true_loss = mseloss(y_pred_noisy, y_true)

            if y_sr is not None:
                loss_sr = mseloss(y_pred_original, y_sr)
                sr_true_loss = mseloss(y_sr, y_true)
                if C.DyOp:
                    original_weight = (sr_true_loss + noisy_true_loss) / (sr_true_loss + orginal_true_loss + noisy_true_loss)
                    noisy_weight = (sr_true_loss + orginal_true_loss) / (sr_true_loss + orginal_true_loss + noisy_true_loss)
                    sr_weight = (orginal_true_loss + noisy_true_loss) / (sr_true_loss + orginal_true_loss + noisy_true_loss)

                    total_loss = 0.5 * original_weight * loss_original + 0.5 * noisy_weight * loss_pred + 0.5 * sr_weight * loss_sr
                    if weight:
                        print(
                            f'original_weight: {original_weight * 0.5}, noisy_weight: {noisy_weight * 0.5}, sr_weight: {sr_weight * 0.5}')
                else:
                    total_loss = 0.33 * loss_original + 0.33 * loss_pred + 0.33 * loss_sr
            else:
                original_weight = noisy_true_loss / (orginal_true_loss + noisy_true_loss)
                noisy_weight = orginal_true_loss / (orginal_true_loss + noisy_true_loss)

                total_loss = original_weight * loss_original + noisy_weight * loss_pred
                if weight:
                    print(f'original_weight: {original_weight}, noisy_weight: {noisy_weight}')
            return total_loss

        else:
            y_pred_original = y_pred_original[:, -1, :]

            loss_original = bceloss(y_pred_original, y_true)
            orginal_true_loss = mseloss(y_pred_original, y_true)

            if y_sr is not None:
                loss_sr = mseloss(y_pred_original, y_sr)
                sr_true_loss = mseloss(y_sr, y_true)
                if C.DyOp:
                    original_weight = (sr_true_loss) / (
                                sr_true_loss + orginal_true_loss)
                    sr_weight = (orginal_true_loss) / (
                                sr_true_loss + orginal_true_loss)

                    total_loss =  original_weight * loss_original + sr_weight * loss_sr
                    if weight:
                        print(
                            f'original_weight: {original_weight}, sr_weight: {sr_weight}')
                else:
                    total_loss = 0.5 * loss_original + 0.5 * loss_sr
            else:
                total_loss = loss_original
            return total_loss

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
    ground_truth = ground_truth.detach().cpu()
    prediction = prediction.detach().cpu()[:, -1, :].squeeze(1)
    if 'momo' in C.DATASET:
        prediction = prediction.unsqueeze(1)
    mae = mean_absolute_error(ground_truth, prediction)
    rmse = root_mean_squared_error(ground_truth, prediction)
    rmsebin = compute_rmse_in_bins(prediction, ground_truth)
    mse = mean_squared_error(ground_truth, prediction)

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
    result = f'MAE: {mae}, RMSE: {rmse}, RMSE(bin): {rmsebin}, AUC: {AUC}, PrAUC: {prauc}, MSE: {mse}, ACC: {acc}, F1: {F1}, recall: {recall}, ' \
             f'precision: {precision} Specificity: {Specificity}, Sensitivity: {Sensitivity}, G_Mean: {G_Mean}\n'
    print(result)
    with open(C.Dpath + C.DATASET + '/DNN-epoch.txt', 'a+', encoding='utf-8') as f:
        f.write(result)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0,
                 save_path=C.Dpath + C.DATASET + '/' + C.Torch_model_name + '_checkpoint.pkl'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.best_model = None
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
        self.best_model = model.state_dict()
        torch.save(self.best_model, self.save_path)
        self.val_loss_min = val_loss

    def load_checkpoint(self):
        model = torch.load(self.save_path)
        return model

# Updated training function with early stopping
def train(trainLoaders, val_loader, model, optimizer, early_stopping, epoch, weight=False):
    model.train()
    epoch_loss = 0
    count = 0
    Data = torch.Tensor([]).to('cuda')
    l = torch.Tensor([]).to('cuda')
    for batch_idx, (data, label) in enumerate(trainLoaders):
        data = data.to('cuda')
        label = label.to('cuda')
        l2_lambda = 0.0001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        pred_original, pred_noisy = model(data)
        pred_original_mask, pred_noisy_mask = pred_original, pred_noisy
        if batch_idx == len(trainLoaders)-1:
            loss = Loss(label.cpu(), pred_original_mask.cpu(), pred_noisy_mask.cpu(), weight=weight) + l2_lambda * l2_norm
        else:
            loss = Loss(label.cpu(), pred_original_mask.cpu(), pred_noisy_mask.cpu()) + l2_lambda * l2_norm

        Data = torch.cat((Data, data), dim=0)
        l = torch.cat((l, label), dim=0)
        optimizer.zero_grad()
        epoch_loss += loss.item()
        count += 1
        loss.backward(retain_graph=True)
        optimizer.step()
    average_loss = epoch_loss / len(trainLoaders)
    print(f'Training Loss: {average_loss:.4f}')

    # Validate the model and check early stopping
    model, optimizer, early_stop = val(val_loader, model, optimizer, early_stopping)
    print(f'Validation loss: {early_stopping.val_loss_min:.4f}')
    if early_stop:
        print("Early stopping")

    model.load_state_dict(early_stopping.best_model)
    if C.Training_model == 'Asy-11' or C.Training_model == 'Asy-00':
        num_samples = 1024
        indices = torch.randperm(Data.size(0))[:num_samples]  # 生成一个随机排列并取前1024个索引
        Data = Data[indices]  # 使用随机索引抽取样本
        l = l[indices]
        pred_original, pred_noisy = model(Data)

        save_DKT_pred(Data, l, pred_original)
        if os.path.exists(C.Dpath + C.DATASET + '/' +C.modelname +'GPSR_pred.npy'):
            SRpred = torch.load(C.Dpath + C.DATASET + '/' +C.modelname +'GPSR_pred.npy')
            model.train()
            for batch_idx, (data, label, srpred) in enumerate(SRpred):
                srpred = torch.Tensor(srpred).float()
                pred_original, pred_noisy = model(data.to('cuda'))
                if batch_idx == len(SRpred)-1:
                    loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu(), weight=weight)
                else:
                    loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

    if C.Training_model == 'Asy-21' and epoch % 2 == 0:
        num_samples = 1024
        indices = torch.randperm(Data.size(0))[:num_samples]  # 生成一个随机排列并取前1024个索引
        Data = Data[indices]  # 使用随机索引抽取样本
        l = l[indices]
        pred_original, pred_noisy = model(Data)
        save_DKT_pred(Data, l, pred_original)
        if os.path.exists(C.Dpath + C.DATASET + '/' +C.modelname +'GPSR_pred.npy'):
            SRpred = torch.load(C.Dpath + C.DATASET + '/' +C.modelname +'GPSR_pred.npy')
            model.train()
            for batch_idx, (data, label, srpred) in enumerate(SRpred):
                srpred = torch.Tensor(srpred).float()
                pred_original, pred_noisy = model(data.to('cuda'))
                if batch_idx == len(SRpred) - 1:
                    loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu(), weight=weight)
                else:
                    loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

    return model, optimizer



# Updated validation function
def val(val_loader, model, optimizer, early_stopping):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to('cuda')
            label = label.to('cuda')
            pred_original, pred_noisy = model(data)
            loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu()).item()
            val_loss += loss
    val_loss /= len(val_loader)
    # Check early stopping condition
    early_stopping(val_loss, model)

    return model, optimizer, early_stopping.early_stop


# Test function remains the same
def test(testLoaders, model, last=False):
    model.eval()
    ground_truth = torch.Tensor([]).to('cuda')
    prediction = torch.Tensor([]).to('cuda')
    loss = 0
    count = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testLoaders):
            data = data.to('cuda')
            label = label.to('cuda')
            pred_original, pred_noisy = model(data)
            loss += Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), None).item()
            count += 1
            prediction = torch.cat([prediction, pred_original.detach()])
            ground_truth = torch.cat([ground_truth, label.detach()])

    # 获取预测值中的最大值
    max_prediction = prediction.max()
    min_prediction = prediction.min()
    average_loss = loss / count
    print(f'Loss: {average_loss:.4f}')
    print(f'Max prediction: {max_prediction}')
    print(f'Min prediction: {min_prediction}')
    performance(ground_truth, prediction)
    ones_tensor = torch.ones_like(prediction)
    zeros_tensor = torch.zeros_like(prediction)
    print('One data _____________________: ')
    performance(ground_truth, ones_tensor)
    print('Zero data _____________________: ')
    performance(ground_truth, zeros_tensor)
    if last == True:
        return ground_truth, prediction



def save_epoch(epoch, ground_truth, prediction):
    if "momo" in C.DATASET:
        ground_truth = ground_truth.detach().cpu()[:, -1].unsqueeze(-1)
    prediction = prediction.detach().cpu()[:, -1, :]

    mae = mean_absolute_error(ground_truth, prediction)
    rmse = root_mean_squared_error(ground_truth, prediction)
    rmsebin = compute_rmse_in_bins(prediction, ground_truth)
    mse = mean_squared_error(ground_truth, prediction)

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

    result = f'epoch: {epoch}  MAE: {mae}, RMSE: {rmse}, RMSE(bin): {rmsebin}, AUC: {AUC}, PrAUC: {prauc}, MSE: {mse}, ACC: {acc}, F1: {F1}, recall: {recall}, precision: {precision}\n'
    with open(C.Dpath + C.DATASET + '/DNN-epoch.txt', 'a+', encoding='utf-8') as f:
        f.write(result)
