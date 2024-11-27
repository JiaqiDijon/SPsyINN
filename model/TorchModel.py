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

# DKT-F model
class DKT_F(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT_F, self).__init__()
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
        self.noise_module = DF_block(input_dim, hidden_dim)
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

        # original data
        original, (h0, c0) = self.rnn(x, (h0, c0))
        original = self.fc(original)  # 64
        out_original = self.out(original)
        h = h0[-1:, :, :].permute(1, 0, 2)
        out_original = self.Linear(out_original + h)
        out_original = self.sig(out_original)

        # noisy data
        noisy, (h0, c0) = self.rnn(noisy_data, (h0, c0))
        noisy = self.fc(noisy)
        out_noisy = self.out(noisy)
        h = h0[-1:, :, :].permute(1, 0, 2)
        out_noisy = self.Linear(out_noisy + h)
        out_noisy = self.sig(out_noisy)

        return out_original, out_noisy

# Loss Function
def Loss(y_true, y_pred_original, y_pred_noisy=None, y_sr=None, weight=False):
    maeloss = nn.L1Loss()
    mseloss = nn.MSELoss()
    if "MaiMemo" in C.DATASET:
        y_true = y_true[:, -1].unsqueeze(-1)
    if C.Torch_model_name in ['DKT-F']:
        y_pred_original = y_pred_original[:, -1, :]
        if y_sr is not None:
            loss_sr = mseloss(y_pred_original, y_sr)
            loss_original = mseloss(y_pred_original, y_true)
            if C.DAO:
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

    elif C.Torch_model_name == 'DNN':
        y_pred_original = y_pred_original[:, -1, :]
        y_pred_noisy = y_pred_noisy[:, -1, :]

        loss_original = mseloss(y_pred_original, y_true)
        loss_pred = mseloss(y_pred_original, y_pred_noisy)

        orginal_true_loss = maeloss(y_pred_original, y_true)
        noisy_true_loss = maeloss(y_pred_noisy, y_true)

        if y_sr is not None:
            loss_sr = mseloss(y_pred_original, y_sr)
            sr_true_loss = maeloss(y_sr, y_true)
            if C.DAO:
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

# MAPE
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

# MAE
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

# Model Evaluation
def performance(ground_truth, prediction):
    ground_truth = ground_truth.detach().cpu()
    if "MaiMemo" in C.DATASET:
        ground_truth = ground_truth[:, -1].unsqueeze(-1)
    prediction = prediction.detach().cpu()[:, -1, :]
    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    print(f'{C.Torch_model_name}:   MAE:' + str(mae) + ' MAPE: ' + str(mape) + '\n')


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

# training
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
        if C.Torch_model_name == 'DKT-F':
            pred_original = model(data)
            loss = Loss(label.cpu(), pred_original.cpu()) + l2_lambda * l2_norm
        if C.Torch_model_name == 'DNN':
            pred_original, pred_noisy = model(data)
            pred_original_mask, pred_noisy_mask = pred_original, pred_noisy
            if batch_idx == len(trainLoaders) - 1:
                loss = Loss(label.cpu(), pred_original_mask.cpu(), pred_noisy_mask.cpu(),
                            weight=weight) + l2_lambda * l2_norm
            else:
                loss = Loss(label.cpu(), pred_original_mask.cpu(), pred_noisy_mask.cpu()) + l2_lambda * l2_norm

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
    model, optimizer, early_stop = val(val_loader, model, optimizer, early_stopping)
    print(f'Validation loss: {early_stopping.val_loss_min:.4f}')

    if early_stop:
        print("Early stopping")

    model.load_state_dict(early_stopping.best_model)
    # DAO:  -W  or -C
    if C.Training_model == 'SPsyINN-W' or C.Training_model == 'SPsyINN-C':
        num_samples = 1024
        indices = torch.randperm(Data.size(0))[:num_samples]  

        Data = Data[indices]  
        l = l[indices]
        if C.Torch_model_name == 'DNN':
            pred_original, pred_noisy = model(Data)
        if C.Torch_model_name == 'DKT-F':
            pred_original = model(Data)
        save_DKT_pred(Data, l, pred_original)
        if os.path.exists(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GSR_pred.npy'):
            SRpred = torch.load(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GSR_pred.npy')
            model.train()
            for batch_idx, (data, label, srpred) in enumerate(SRpred):
                srpred = torch.Tensor(srpred).float()
                if C.Torch_model_name == 'DNN':
                    pred_original, pred_noisy = model(data.to('cuda'))
                    if batch_idx == len(SRpred) - 1:
                        loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu(), weight=weight)
                    else:
                        loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu())

                if C.Torch_model_name == 'DKT-F':
                    pred_original = model(data.to('cuda'))
                    loss = Loss(label.cpu(), pred_original.cpu(), y_pred_noisy=None, y_sr=srpred.cpu())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
    # DAO:  -I
    if C.Training_model == 'SPsyINN-I' and epoch % 2 == 0:
        num_samples = 1024
        indices = torch.randperm(Data.size(0))[:num_samples]  

        Data = Data[indices]  
        l = l[indices]
        if C.Torch_model_name == 'DNN':
            pred_original, pred_noisy = model(Data)
        if C.Torch_model_name == 'DKT-F':
            pred_original = model(Data)

        save_DKT_pred(Data, l, pred_original)
        if os.path.exists(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GSR_pred.npy'):
            SRpred = torch.load(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'GSR_pred.npy')
            model.train()
            for batch_idx, (data, label, srpred) in enumerate(SRpred):
                srpred = torch.Tensor(srpred).float()
                if C.Torch_model_name == 'DNN':
                    pred_original, pred_noisy = model(data.to('cuda'))
                    if batch_idx == len(SRpred) - 1:
                        loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu(), weight=weight)
                    else:
                        loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), srpred.cpu())
                if C.Torch_model_name == 'DKT-F':
                    pred_original = model(data.to('cuda'))
                    loss = Loss(label.cpu(), pred_original.cpu(), y_pred_noisy=None, y_sr=srpred.cpu())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

    return model, optimizer


# val for early stop
def val(val_loader, model, optimizer, early_stopping):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to('cuda')
            label = label.to('cuda')
            if C.Torch_model_name == 'DKT-F':
                pred_original = model(data)
            if C.Torch_model_name == 'DNN':
                pred_original, pred_noisy = model(data)

            if C.Torch_model_name == 'DKT-F':
                loss = Loss(label.cpu(), pred_original.cpu()).item()
            if C.Torch_model_name == 'DNN':
                loss = Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu()).item()
            val_loss += loss

    val_loss /= len(val_loader)

    # Check early stopping condition
    early_stopping(val_loss, model)

    return model, optimizer, early_stopping.early_stop


# Test
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
            if C.Torch_model_name == 'DKT-F':
                pred_original = model(data)
                loss += Loss(label.cpu(), pred_original.cpu()).item()
            if C.Torch_model_name == 'DNN':
                pred_original, pred_noisy = model(data)
                loss += Loss(label.cpu(), pred_original.cpu(), pred_noisy.cpu(), None).item()
            count += 1
            prediction = torch.cat([prediction, pred_original.detach()])
            ground_truth = torch.cat([ground_truth, label.detach()])

    performance(ground_truth, prediction)
    if last == True:
        return ground_truth.cpu(), prediction.cpu()

# save result
def save_epoch(epoch, ground_truth, prediction):
    if "MaiMemo" in C.DATASET:
        ground_truth = ground_truth.detach().cpu()[:, -1].unsqueeze(-1)
    prediction = prediction.detach().cpu()[:, -1, :]
    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    result = f'{epoch} :   MAE: {mae.item()} MAPE: {mape.item()}\n'
    with open(C.Dpath + C.DATASET + '/DNN-epoch.txt', 'a+', encoding='utf-8') as f:
        f.write(result)
