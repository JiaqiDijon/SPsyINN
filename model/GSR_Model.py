import os
import torch
import pickle
import numpy as np
import model.Constants as C
from model.Sample_data import save_PYSR_pred
import warnings
warnings.filterwarnings("ignore")

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
    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    result = f'GSR:   MAE: {mae.item()} MAPE: {mape.item()}\n'
    result += '----------------------------------------------------------------------------------\n'
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


def train(model, trainLoaders, early_stopping, valLoaders):
    X = torch.Tensor([])
    y = torch.Tensor([])
    Data = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(trainLoaders):
        X = torch.cat([X, data[:, -1, :]], dim=0)
        y = torch.cat([y, label], dim=0)
        Data = torch.cat([Data, data], dim=0)

    num_samples = 1024
    indices = torch.randperm(X.size(0))[:num_samples]
    X = X[indices]
    y_sr = y[indices]
    model.fit(X.detach().numpy(), y_sr.detach().numpy())

    model, val_loss = val(model, valLoaders)

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
    model = early_stopping.load_checkpoint()
    indices_forDNN = torch.randperm(Data.size(0))[:num_samples]
    Data = Data[indices_forDNN]  
    y = y[indices_forDNN]


    Data = torch.nan_to_num(Data, 0)
    pred = model.predict(Data[:, -1, :])

    save_PYSR_pred(Data, y, pred.reshape(pred.shape[0], 1))
    if os.path.exists(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'DNN_pred.npy'):
        D = torch.Tensor([])
        L = torch.Tensor([])
        DKTpred = torch.load(C.Dpath + C.DATASET + '/' + C.Torch_model_name + 'DNN_pred.npy')
        for batch_idx, (data, label, dktpred) in enumerate(DKTpred):
            D = torch.cat([D, data[:, -1, :].cpu()], dim=0)
            L = torch.cat([L, dktpred[:, -1, :].cpu()], dim=0)
        model.fit(D.cpu().detach().numpy(), L.cpu().detach().numpy())

    return model


def val(model, valLoaders):
    X = torch.Tensor([])
    y = torch.Tensor([])
    prediction = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(valLoaders):
        X = torch.cat([X, data], dim=0)
        y = torch.cat([y, label], dim=0)
        data = data[:, -1, :]
        pred = torch.Tensor([model.predict(data.cpu().detach().numpy())]).squeeze(0).unsqueeze(1)
        prediction = torch.cat([prediction, pred], dim=0)

    val_loss = torch.nn.functional.mse_loss(prediction, y)
    return model, val_loss.item()


def test(model, testLoaders, last=False):
    ground_truth = torch.Tensor([])
    prediction = torch.Tensor([])
    for batch_idx, (data, label) in enumerate(testLoaders):
        data = data[:, -1, :]
        ground_truth = torch.cat([ground_truth, label.cpu()])
        SR_pred = torch.Tensor([model.predict(data.cpu().detach().numpy())])
        prediction = torch.cat([prediction, SR_pred.squeeze(0).unsqueeze(1)], dim=0)

    performance(ground_truth, prediction)
    if last == True:
        return ground_truth, prediction

