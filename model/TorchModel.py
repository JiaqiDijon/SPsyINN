import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.Sample_data import save_DKT_pred
import model.Constants as C
import warnings
import math
from torch.nn import functional as F

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True
CUDA_LAUNCH_BLOCKING = 1


class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 64)
        self.out = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        self.Linear = nn.Linear(64, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)).to(x.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)).to(x.device)
        rnn_out, (h0, c0) = self.rnn(x, (h0, c0))
        out = self.fc(rnn_out)
        out = self.out(out)
        h = h0[-1:, :, :].permute(1, 0, 2)
        out = self.Linear(out + h)
        out = self.sig(out)

        return out


class DF_block(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, noise_steps=100, beta_start=1e-3, beta_end=0.2,
                 dropout_prob=0.2):
        super(DF_block, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # 定义去噪模型
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.denoise_model = nn.Sequential(*layers)

        # 噪声水平作为可学习参数
        self.noise_level = nn.Parameter(torch.tensor(0.5))

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alpha_hat[t].to(x.device)
        alpha_t_sqrt = torch.sqrt(alpha_t).view(-1, 1, 1)
        noise_t = noise * torch.sqrt(1 - alpha_t).view(-1, 1, 1)

        # 根据可学习的噪声水平调整噪声强度
        noise_t = noise_t * self.noise_level

        noisy_data = alpha_t_sqrt * x + noise_t
        return noisy_data.squeeze(0), noise

    def denoise(self, noisy_data, t):
        alpha_t = self.alpha_hat[t].to(noisy_data.device)
        alpha_t_sqrt = torch.sqrt(alpha_t).view(-1, 1, 1)
        noise_pred = self.denoise_model(noisy_data / alpha_t_sqrt)
        return (noisy_data - noise_pred * torch.sqrt(1 - alpha_t).view(-1, 1, 1)) / alpha_t_sqrt


class DTKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DTKT, self).__init__()
        self.noise_module = DF_block(input_dim, hidden_dim, num_layers=5)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(64, 64)
        self.out = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        self.Linear = nn.Linear(64, self.output_dim)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.t = 3

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)).to(x.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim)).to(x.device)
        noisy_data, noise = self.noise_module(x, t=15)

        # 对原始数据的预测
        original, (h0, c0) = self.rnn(x, (h0, c0))
        original = self.fc(original)  # 64
        out_original = self.out(original)
        h = h0[-1:, :, :].permute(1, 0, 2)
        out_original = self.Linear(out_original + h)
        out_original = self.sig(out_original)

        # 对添加噪声数据的预测
        noisy, (h0, c0) = self.rnn(noisy_data, (h0, c0))
        noisy = self.fc(noisy)
        out_noisy = self.out(noisy)
        h = h0[-1:, :, :].permute(1, 0, 2)
        out_noisy = self.Linear(out_noisy + h)
        out_noisy = self.sig(out_noisy)

        return out_original, out_noisy


class FIFAKT(nn.Module):
    def __init__(self, n_question, p_num, embed_l, embed_p, hidden_dim, input_size, layer_dim=1, class_weights=None,
                 final_fc_dim=512, dropout=0.0, z_weight=0.0, pretrained_embeddings=None, freeze_pretrained=True):
        super(FIFAKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.z_weight = z_weight
        self.class_weights = class_weights
        self.n_question = n_question
        self.p_num = p_num
        self.layer_dim = layer_dim
        self.input_size = input_size

        # pretrained_embeddings=None
        if pretrained_embeddings is not None:
            print("embeddings frozen:", freeze_pretrained, flush=True)
            self.q_embed = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=0, freeze=freeze_pretrained)
        else:
            self.q_embed = nn.Embedding(self.n_question, embed_l, padding_idx=0)
            self.p_embed = nn.Embedding(self.p_num, embed_p, padding_idx=0)
        num_features = self.input_size
        if 'momo' in C.DATASET:
            self.rnn = nn.LSTM(
                input_size=embed_l + embed_p + 6 + num_features,  # 181
                hidden_size=self.hidden_dim,
                num_layers=self.layer_dim,
            )
            self.out = nn.Sequential(
                nn.Linear(270, 64),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(64, 10),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(10, 1),
            )
        else:
            self.rnn = nn.LSTM(
                input_size=embed_l + embed_p + 11 + num_features,  # 186
                hidden_size=self.hidden_dim,
                num_layers=self.layer_dim,
            )
            self.out = nn.Sequential(
                nn.Linear(275, 64),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(64, 10),
                nn.Tanh(),
                nn.Dropout(0.3),
                nn.Linear(10, 1),
            )
        self.sig = nn.Sigmoid()

    def attention_net_q(self, q_context, state, l):
        # hidden = final_state.view(-1, self.hidden_dim , 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        q_context_t = q_context.transpose(1, 2)

        attn_weights_o = torch.bmm(q_context, q_context_t).squeeze(2)  # attn_weights : [batch_size, n_step]

        attn_weights = attn_weights_o[:, :, :]
        scaled_attn_weights = torch.divide(attn_weights, math.sqrt(l))
        scaled_attn_weights = torch.triu(scaled_attn_weights)
        scaled_attn_weights[scaled_attn_weights == 0] = -1000

        soft_attn_weights = F.softmax(scaled_attn_weights, dim=-2)
        soft_attn_weights = torch.triu(soft_attn_weights)
        context = torch.bmm(state.transpose(1, 2), soft_attn_weights).squeeze(2)

        context = context.transpose(1, 2)

        return context, soft_attn_weights.data

    def forward(self, data):

        data_x, data_y = data[:-1, :, :], data[-1, :, :]

        # word embeding
        q_data = data_x[:, :, 4].unsqueeze(-1)
        q_embed_data = self.q_embed(
            q_data.to(dtype=torch.long)).squeeze()  # input : [batch_size, len_seq, embedding_dim]
        # user embeding
        p_data = data_x[:, :, 0].unsqueeze(-1)
        p_embed_data = self.p_embed(p_data.to(dtype=torch.long)).squeeze()
        batch_size, sl = data_x.size(1), data_x.size(0)
        hidden_state = Variable(torch.zeros(self.layer_dim, batch_size, self.hidden_dim,
                                            device=device))  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(self.layer_dim, batch_size, self.hidden_dim,
                                          device=device))  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        rnn_input = torch.cat([q_embed_data, p_embed_data, data_x, data_y.unsqueeze(0).repeat(data_x.shape[0], 1, 1)],
                              dim=2)

        output, (final_hidden_state, final_cell_state) = self.rnn(rnn_input, (hidden_state, cell_state))

        att_input = torch.cat([q_embed_data, p_embed_data], dim=2)
        output = output
        attn_output, attention = self.attention_net_q(att_input, output, l=len(att_input))
        ffn_input = torch.cat([attn_output, output[:, :, :], p_embed_data[:, :, :], q_embed_data[:, :, :],
                               data_y[:, :].unsqueeze(0).repeat(data_x.shape[0], 1, 1)], dim=2)
        ffn_input = ffn_input.transpose(0, 1)
        ffn_input = ffn_input[:, -1, :]

        pred = self.out(ffn_input)
        # pred = self.sig(pred)
        return pred


def Loss(y_true, y_pred_original, y_pred_noisy=None, y_sr=None, weight=False):
    maeloss = nn.L1Loss()
    mseloss = nn.MSELoss()

    if C.Torch_model_name in ['DKT']:
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

    elif C.Torch_model_name == 'DTKT':
        y_pred_original = y_pred_original[:, -1, :]
        y_pred_noisy = y_pred_noisy[:, -1, :]

        loss_original = mseloss(y_pred_original, y_true)
        loss_pred = mseloss(y_pred_original, y_pred_noisy)

        orginal_true_loss = maeloss(y_pred_original, y_true)
        noisy_true_loss = maeloss(y_pred_noisy, y_true)

        if y_sr is not None:
            loss_sr = mseloss(y_pred_original, y_sr)
            sr_true_loss = maeloss(y_sr, y_true)
            if C.DyOp:
                original_weight = (sr_true_loss + noisy_true_loss) / (
                            sr_true_loss + orginal_true_loss + noisy_true_loss)
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

    elif C.Torch_model_name == 'FIFKT':
        if y_sr is not None:
            loss_sr = mseloss(y_pred_original, y_sr)
            loss_original = mseloss(y_pred_original, y_true)

            orginal_true_loss = maeloss(y_pred_original, y_true)
            sr_true_loss = maeloss(y_sr, y_true)

            original_weight = sr_true_loss / (sr_true_loss + orginal_true_loss)
            sr_weight = orginal_true_loss / (sr_true_loss + orginal_true_loss)

            total_loss = original_weight * loss_original + sr_weight * loss_sr
            if weight:
                print(f'original_weight: {original_weight}, sr_weight: {sr_weight}')
        else:
            total_loss = mseloss(y_pred_original, y_true)
        return total_loss


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


def performance(ground_truth, prediction):
    ground_truth = ground_truth.detach().cpu()
    prediction = prediction.detach().cpu()

    if C.Torch_model_name != 'FIFKT':
        prediction = prediction[:, -1, :]

    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    mse = masked_mse(prediction, ground_truth)
    rmse = masked_rmse(prediction, ground_truth)

    # results.updata(mae.detach(), mse.detach(), mape.detach(), rmse.detach())

    print('  DKT:   MAE:' + str(mae) + ' MAPE: ' + str(mape) + ' MSE: ' + str(mse) + ' RMSE: ' + str(rmse) + '\n')

    result = '  DKT:   MAE:' + str(mae) + ' MAPE: ' + str(mape) + ' MSE: ' + str(mse) + ' RMSE: ' + str(rmse) + '\n' + \
             '----------------------------------------------------------------------------------\n'

    with open(C.Dpath + C.DATASET + '/DKTresult.txt', 'a+', encoding='utf-8') as f:
        f.write(result)

    # return results


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
    X = torch.tensor([])

    for batch_idx, (data, label) in enumerate(trainLoaders):
        X = torch.cat((X, data), dim=0)

    min = torch.min(X.reshape(-1, C.INPUT), dim=0).values
    max = torch.max(X.reshape(-1, C.INPUT), dim=0).values

    del X

    model.train()
    epoch_loss = 0
    count = 0
    Data = torch.Tensor([]).to('cuda')
    l = torch.Tensor([]).to('cuda')
    for batch_idx, (data, label) in enumerate(trainLoaders):
        data = (data - min) / (max - min)
        data = data.to('cuda')
        label = label.to('cuda')
        l2_lambda = 0.0001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        if C.Torch_model_name == 'DKT' or C.Torch_model_name == 'TKT':
            pred_original = model(data)
            loss = Loss(label.cpu(), pred_original.cpu()) + l2_lambda * l2_norm
        if C.Torch_model_name == 'DTKT':
            pred_original, pred_noisy = model(data)
            pred_original_mask, pred_noisy_mask = pred_original, pred_noisy
            if batch_idx == len(trainLoaders) - 1:
                loss = Loss(label.cpu(), pred_original_mask.cpu(), pred_noisy_mask.cpu(),
                            weight=weight) + l2_lambda * l2_norm
            else:
                loss = Loss(label.cpu(), pred_original_mask.cpu(), pred_noisy_mask.cpu()) + l2_lambda * l2_norm
        if C.Torch_model_name == 'FIFKT':
            data = data.transpose(0, 1)
            pred = model(data)
            loss = Loss(label.cpu().squeeze(), pred[0].cpu().squeeze())
            data = data.transpose(0, 1)

        Data = torch.cat((Data, data), dim=0)
        l = torch.cat((l, label), dim=0)
        optimizer.zero_grad()
        epoch_loss += loss.item()
        count += 1
        loss.backward(retain_graph=True)
        optimizer.step()
    average_loss = epoch_loss / count
    print(f'Training Loss: {average_loss:.4f}')

    # Validate the model and check early stopping
    model, optimizer, early_stop = val(val_loader, model, optimizer, early_stopping, min, max)
    print(f'Validation loss: {early_stopping.val_loss_min:.4f}')

    if early_stop:
        print("Early stopping")

    model.load_state_dict(early_stopping.best_model)

    if C.Training_model == 'Asy-11' or C.Training_model == 'Asy-00':
        num_samples = 1024
        indices = torch.randperm(Data.size(0))[:num_samples]  # 生成一个随机排列并取前1024个索引

        Data = Data[indices]  # 使用随机索引抽取样本
        l = l[indices]
        if C.Torch_model_name == 'DTKT':
            pred_original, pred_noisy = model(Data)
        if C.Torch_model_name == 'DKT':
            pred_original = model(Data)
        if C.Torch_model_name == 'FIFKT':
            Data = Data.transpose(0, 1)
            pred_original = model(Data)
            Data = Data.transpose(0, 1)

        save_DKT_pred(Data, l, pred_original)
        if os.path.exists(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GPSR_pred.npy'):
            SRpred = torch.load(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GPSR_pred.npy')
            model.train()
            for batch_idx, (data, label, srpred) in enumerate(SRpred):
                srpred = torch.Tensor(srpred).float()
                if C.Torch_model_name == 'DTKT':
                    pred_original, pred_noisy = model(data.to('cuda'))
                    if batch_idx == len(SRpred) - 1:
                        loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu(), weight=weight)
                    else:
                        loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu())

                if C.Torch_model_name == 'FIFKT':
                    data = data.transpose(0, 1).to('cuda')
                    pred_original = model(data)
                    data = data.transpose(0, 1)
                    if batch_idx == len(SRpred) - 1:
                        loss = Loss(label.cpu(), pred_original.cpu(), y_pred_noisy=None, y_sr=srpred.cpu(),
                                    weight=weight)
                    else:
                        loss = Loss(label.cpu(), pred_original.cpu(), y_pred_noisy=None, y_sr=srpred.cpu())

                if C.Torch_model_name == 'DKT':
                    pred_original = model(data.to('cuda'))
                    loss = Loss(label.cpu(), pred_original.cpu(), y_pred_noisy=None, y_sr=srpred.cpu())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

    if C.Training_model == 'Asy-21' and epoch % 2 == 0:
        num_samples = 1024
        indices = torch.randperm(Data.size(0))[:num_samples]  # 生成一个随机排列并取前1024个索引

        Data = Data[indices]  # 使用随机索引抽取样本
        l = l[indices]
        if C.Torch_model_name == 'DTKT':
            pred_original, pred_noisy = model(Data)
        if C.Torch_model_name == 'DKT':
            pred_original = model(Data)
        if C.Torch_model_name == 'FIFKT':
            Data = Data.transpose(0, 1).to('cuda')
            pred_original = model(Data)

        save_DKT_pred(Data, l, pred_original)
        if os.path.exists(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GPSR_pred.npy'):
            SRpred = torch.load(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GPSR_pred.npy')
            model.train()
            for batch_idx, (data, label, srpred) in enumerate(SRpred):
                srpred = torch.Tensor(srpred).float()
                if C.Torch_model_name == 'DTKT':
                    pred_original, pred_noisy = model(data.to('cuda'))
                    if batch_idx == len(SRpred) - 1:
                        loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu(), weight=weight)
                    else:
                        loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu())

                if C.Torch_model_name == 'FIFKT':
                    data = data.transpose(0, 1).to('cuda')
                    pred_original = model(data)
                    data = data.transpose(0, 1)
                    if batch_idx == len(SRpred) - 1:
                        loss = Loss(label.cpu(), pred_original.cpu(), y_pred_noisy=None, y_sr=srpred.cpu(),
                                    weight=weight)
                    else:
                        loss = Loss(label.cpu(), pred_original.cpu(), y_pred_noisy=None, y_sr=srpred.cpu())

                if C.Torch_model_name == 'DKT':
                    pred_original = model(data.to('cuda'))
                    loss = Loss(label.cpu(), pred_original.cpu(), y_pred_noisy=None, y_sr=srpred.cpu())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

    return model, optimizer, min, max


# Updated validation function
def val(val_loader, model, optimizer, early_stopping, min, max):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = (data - min) / (max - min)
            data = data.to('cuda')
            label = label.to('cuda')
            if C.Torch_model_name == 'DKT' or C.Torch_model_name == 'TKT':
                pred_original = model(data)
            if C.Torch_model_name == 'DTKT':
                pred_original, pred_noisy = model(data)
            if C.Torch_model_name == 'FIFKT':
                data = data.transpose(0, 1)
                pred = model(data)
                data = data.transpose(0, 1)

            if C.Torch_model_name == 'DKT' or C.Torch_model_name == 'TKT':
                loss = Loss(label.cpu(), pred_original.cpu()).item()
            if C.Torch_model_name == 'DTKT':
                loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu()).item()
            if C.Torch_model_name == 'FIFKT':
                loss = Loss(label.cpu(), pred.cpu()).item()

            val_loss += loss

    val_loss /= len(val_loader)

    # Check early stopping condition
    early_stopping(val_loss, model)

    return model, optimizer, early_stopping.early_stop


# Test function remains the same
def test(testLoaders, model, min, max, last=False):
    model.eval()
    ground_truth = torch.Tensor([]).to('cuda')
    prediction = torch.Tensor([]).to('cuda')
    loss = 0
    count = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testLoaders):
            data = (data - min) / (max - min)
            data = data.to('cuda')
            label = label.to('cuda')
            if C.Torch_model_name == 'DKT' or C.Torch_model_name == 'TKT':
                pred_original = model(data)
                loss += Loss(label.cpu(), pred_original.cpu()).item()
            if C.Torch_model_name == 'DTKT':
                pred_original, pred_noisy = model(data)
                loss += Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), None).item()
            if C.Torch_model_name == 'FIFKT':
                data = data.transpose(0, 1)
                pred_original = model(data)
                loss += Loss(label.cpu(), pred_original.cpu()).item()

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
    if last == True:
        return ground_truth, prediction


def save_epoch(epoch, ground_truth, prediction):
    ground_truth = ground_truth.detach().cpu()
    prediction = prediction.detach().cpu()
    if C.Torch_model_name != 'FIFKT':
        prediction = prediction[:, -1, :]

    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    mse = masked_mse(prediction, ground_truth)
    rmse = masked_rmse(prediction, ground_truth)
    result = f'{epoch} :   MAE: {mae.item()} MAPE: {mape.item()} MSE: {mse.item()} RMSE: {rmse.item()}\n'
    with open(C.Dpath + C.DATASET + '/DDKT-epoch.txt', 'a+', encoding='utf-8') as f:
        f.write(result)
