import torch
import warnings
import numpy as np
import torch.nn as nn

from data_loader import getLoaders, CustomNormalizedDataLoader, MyDataste

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


class HLR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.n_input = input_dim
        self.n_out = 1
        self.fc = nn.Linear(self.n_input, self.n_out)

    def forward(self, x):
        t = x[:, :, 1].unsqueeze(2)
        h = self.fc(x)
        recall = torch.pow(0.5, abs(t / h))
        return recall[:, -1, :]

    def test(self, x, name):
        if name == 'Duolingo':
            t = x[:, :, 1]
            h = x @ torch.tensor([3.5303, -9.5382, -0.2006, -0.0419, -0.1819, 0.3865]).to('cuda') - 0.0660
            recall = torch.pow(0.5, abs(t / h))[:, -1].unsqueeze(1)
        if name == 'MaiMemo':
            t = x[:, :, 1]
            h = x @ torch.tensor([2.5871, -4.3821, -0.4100, 0.3737, 0.0108]).to('cuda') - 0.0469
            recall = torch.pow(0.5, abs(t / h))[:, -1].unsqueeze(1)

        return recall


class Wick(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.n_input = input_dim
        self.n_out = 1
        self.w = nn.Parameter(torch.tensor([0.17, 0.21, 0.15], dtype=torch.float32))

    def forward(self, x):
        fa = self.w[2]
        recall = self.w[0] * torch.pow((1 + self.w[1] * x[:, :, 1]).unsqueeze(2), -fa)[:, -1]
        return recall

    def test(self, x, name):
        if name == 'Duolingo':
            fa = torch.tensor([-0.0003]).to('cuda')
            recall = 0.89 * torch.pow((1 + 0.00031 * x[:, :, 1]), -fa)[:, -1].unsqueeze(1)  # OK
        if name == 'MaiMemo':
            fa = torch.tensor([-0.114]).to('cuda')
            recall = 0.6494 * torch.pow((1 + 21.2314 * x[:, :, 1]), -fa)[:, -1].unsqueeze(1)  # OK

        return recall


class ACT_R(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.n_input = input_dim
        self.n_out = 1
        self.w = nn.Parameter(torch.tensor([0.17, 0.21], dtype=torch.float32))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b = self.w[0]
        d = self.w[1]
        s = torch.sum(torch.pow(x[:, :, 1].clamp_min(1e-7), d.squeeze()), dim=1)
        recall = b + torch.log(s)
        recall = self.sig(recall).unsqueeze(1)
        return recall

    def test(self, x, name):
        if name == 'Duolingo':
            b = torch.tensor([-0.5591]).to('cuda')
            d = torch.tensor([0.0482]).to('cuda')
            s = torch.sum(torch.pow(x[:, :, 1].clamp_min(1e-7), d), dim=1)
            recall = b + torch.log(s)
            recall = self.sig(recall).unsqueeze(1)  # OK
        if name == 'MaiMemo':
            b = torch.tensor([0.8419]).to('cuda')
            d = torch.tensor([0.4319]).to('cuda')
            s = torch.sum(torch.pow(x[:, :, 1].clamp_min(1e-7), d), dim=1)
            recall = b + torch.log(s)
            recall = self.sig(recall).unsqueeze(1)  # OK

        return recall


class DASH(nn.Module):
    def __init__(self, input_dim):
        super(DASH, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.w = nn.Parameter(torch.tensor([0.2783, 0.8131, 0.4252, 1.0056], dtype=torch.float32))

    def forward(self, x):
        a = self.w[0]
        dc = self.w[1]
        d = dc * x[:, :, -1]
        cw = x[:, :, -2]  # correct
        nw = x[:, :, -3]  # seen
        s1 = self.w[2]
        s2 = self.w[3]

        recall = a - d[:, -1] + torch.sum(s1 * torch.log(1 + cw), dim=1) + s2 * torch.log(1 + nw)[:, -1]
        recall = self.sigmoid(recall).unsqueeze(1)

        return recall

    def test(self, x, name):
        if name == 'Duolingo':
            a = torch.tensor([1.8879]).to('cuda')
            dc = torch.tensor([1.1350]).to('cuda')
            d = dc * x[:, :, -1]
            cw = x[:, :, -2]
            nw = x[:, :, -3]
            s1 = torch.tensor([0.1027]).to('cuda')
            s2 = torch.tensor([-0.2019]).to('cuda')
            recall = a - d[:, -1] + torch.sum(s1 * torch.log(1 + cw), dim=1) + s2 * torch.log(1 + nw)[:, -1]  # OK
            recall = self.sigmoid(recall).unsqueeze(1)
        if name == 'MaiMemo':
            a = torch.tensor([0.4186]).to('cuda')
            dc = torch.tensor([0.4267]).to('cuda')
            d = dc * x[:, :, -1]
            cw = x[:, :, -2]
            nw = x[:, :, -3]
            s1 = torch.tensor([0.1790]).to('cuda')
            s2 = torch.tensor([4.0767]).to('cuda')
            recall = a - d[:, -1] + torch.sum(s1 * torch.log(1 + cw), dim=1) + s2 * torch.log(1 + nw)[:, -1]  # OK
            recall = self.sigmoid(recall).unsqueeze(1)
        return recall



class DSR(nn.Module):
    def __init__(self):
        super(DSR).__init__()

    def test(self, x, name):
        # sin(exp(x5*exp(x2-exp(x4))))
        if name == 'Duolingo':
            x2 = x[:, :, 1]  # \delta_2
            x4 = x[:, :, 3]  # \delta_4
            x5 = x[:, :, 4]  # \delta_5
            result = torch.sin(torch.exp(x5 * torch.exp(x2 - torch.exp(x4))))
            recall = result[:, -1].unsqueeze(1)
        # cos(x1 - x4 + exp(x1*x2*x3(x3-x4-x5)+x5))
        if name == 'MaiMemo':
            x1 = x[:, :, 0]  # \delta_1
            x2 = x[:, :, 1]  # \delta_2
            x3 = x[:, :, 2]  # \delta_4
            x4 = x[:, :, 3]  # \delta_5
            x5 = x[:, :, 4]  # \delta_6
            inner_term = x3 * (-x3 - x4 - x5) + x5
            product_term = x1 * x2 * inner_term
            exp_term = torch.exp(product_term)
            result = torch.cos(x1 - x4 + exp_term)

            recall = result[:, -1].unsqueeze(1)
        return recall



class SPsyINN_C(nn.Module):
    def __init__(self):
        super(SPsyINN_C).__init__()

    def test(self, x, name):
        # -(x0 + x2)*(x1 - x2) + 0.9135621262263904
        if name == 'Duolingo':
            x1 = x[:, :, 0]  # \delta_1
            x2 = x[:, :, 1]  # \delta_2
            x6 = x[:, :, 5]  # \delta_6
            result = -(x6 + x2)*(x1 - x2) + 0.913562126226390
            recall = result[:, -1].unsqueeze(1)

        # 0.30229160710443679**((x1*x4)**(x3 + 0.14383100468725032))
        if name == 'MaiMemo':
            x1 = x[:, :, 0]  # \delta_1
            x3 = x[:, :, 2]  # \delta_4
            x4 = x[:, :, 3]  # \delta_5

            result = 0.30229160710443679**((x1*x4)**(x3 + 0.14383100468725032))
            recall = result[:, -1].unsqueeze(1)

        return recall


class SPsyINN_W(nn.Module):
    def __init__(self):
        super(SPsyINN_W).__init__()

    def test(self, x, name):
        # ((x5 + 0.0190478124994636)**(x1-x2))*0.9257066642765628**e**x6
        if name == 'Duolingo':
            x1 = x[:, :, 0]  # \delta_1
            x2 = x[:, :, 1]  # \delta_2
            x5 = x[:, :, 4]  # \delta_5
            x6 = x[:, :, 5]  # \delta_6
            result = torch.pow((x5 + 0.0190478124994636), (x1 - x2))*torch.pow(0.9257066642765628, torch.exp(x6))
            recall = result[:, -1].unsqueeze(1)
        # 0.49258729183071737**((x1 + 0.008248828395980482)**(x3**0.6295170754361378))
        if name == 'MaiMemo':
            x1 = x[:, :, 0]  # \delta_1
            x3 = x[:, :, 2]  # \delta_4
            result = 0.49258729183071737**((x1 + 0.008248828395980482)**(x3**0.6395170754361378))
            recall = result[:, -1].unsqueeze(1)
        return recall


class SPsyINN_I(nn.Module):
    def __init__(self):
        super(SPsyINN_I).__init__()

    def test(self, x, name):
        # 0.92171514151 ** (0.2101281541 * x1 + torch.exp(x6))
        if name == 'Duolingo':
            x1 = x[:, :, 0]  # \delta_1
            x6 = x[:, :, 5]  # \delta_6
            result = 0.92171514151**(0.2101281541*x1 + torch.exp(x6))
            recall = result[:, -1].unsqueeze(1)

        # 0.4733634187963817**((x1**1.109281583119398)**(x3**0.7567579728325409))
        if name == 'MaiMemo':
            x1 = x[:, :, 0]  # \delta_1
            x3 = x[:, :, 2]  # \delta_4
            result = 0.4733634187963817**((x1**1.109281583119398)**(x3**0.7567579728325409))

            recall = result[:, -1].unsqueeze(1)
        return recall


class PySR(nn.Module):
    def __init__(self):
        super(PySR).__init__()

    def test(self, x, name):
        # -exp(x1) + exp(x2) + 0.90370266152383037
        if name == 'Duolingo':
            x1 = x[:, :, 0]  # \delta_1
            x2 = x[:, :, 1]  # \delta_2

            result = -torch.exp(x1) + torch.exp(x2) + 0.90370266152383037
            recall = result[:, -1].unsqueeze(1)

        # 0.2091122905833945**((x1**2)**(x3 + 0.0792623746128449))
        if name == 'MaiMemo':
            x1 = x[:, :, 0]  # \delta_1
            x3 = x[:, :, 2]  # \delta_4
            result = 0.2091122905833945**((x1**2)**(x3 + 0.0792613646118449))
            recall = result[:, -1].unsqueeze(1)

        return recall


class EarlyStopping:
    def __init__(self, function, patience=5, verbose=False, delta=0, ):
        self.patience = patience
        self.function = function
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.best_model = None
        self.save_path = self.function + '_checkpoint.pkl'

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

    mape = masked_mape(prediction, ground_truth)
    mae = masked_mae(prediction, ground_truth)
    mse = masked_mse(prediction, ground_truth)
    rmse = masked_rmse(prediction, ground_truth)

    print('Test : MAE:' + str(mae) + ' MAPE: ' + str(mape) + ' MSE: ' + str(mse) + ' RMSE: ' + str(rmse) + '\n')


def train(model, TrainLoaders, criterion, optimizer, valLoader, early_stopping, input_dim):
    model.train()
    for batch_idx, (data, label) in enumerate(TrainLoaders):
        data = data.to('cuda')
        label = label.to('cuda')
        pred_original = model(data)
        if input_dim == 5:
            label = label[:, -1].unsqueeze(1)
        loss = criterion(label.cpu(), pred_original.cpu())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    model, early_stop = val(valLoader, model, criterion, early_stopping, input_dim)

    if early_stop:
        print("Early stopping")

    model.load_state_dict(torch.load(early_stopping.save_path))

    return model, optimizer


def val(val_loader, model, criterion, early_stopping, input_dim):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        batch = 0
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to('cuda')
            label = label.to('cuda')
            if input_dim == 5:
                label = label[:, -1].unsqueeze(1)
            pred_original = model(data)
            loss = criterion(label.cpu(), pred_original.cpu()).item()
            val_loss += loss
            batch += 1
        # Check early stopping condition
        early_stopping(val_loss / batch, model)
        return model, early_stopping.early_stop


def test(model, test_loader, input_dim):
    model.eval()
    ground_truth = torch.Tensor([]).to('cuda')
    prediction = torch.Tensor([]).to('cuda')
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to('cuda')
            label = label.to('cuda')
            if input_dim == 5:
                label = label[:, -1].unsqueeze(1)
            pred_original = model(data)
            prediction = torch.cat([prediction, pred_original.detach()])
            ground_truth = torch.cat([ground_truth, label.detach()])
        performance(ground_truth, prediction)
        pred_min = prediction.min()
        pred_max = prediction.max()
        print(f'pred min {pred_min}, pred max {pred_max}')


def print_model_param_names(model):
    for name, param in model.named_parameters():
        print(name)


def print_model_param_values(model):
    for name, param in model.named_parameters():
        print(name, param.data)


def test_only(model, test_loader, name, input_dim):
    ground_truth = torch.Tensor([]).to('cuda')
    prediction = torch.Tensor([]).to('cuda')
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to('cuda')
            label = label.to('cuda')
            if input_dim == 5:
                label = label[:, -1].unsqueeze(1)
            pred_original = model.test(data, name)
            prediction = torch.cat([prediction, pred_original.detach().to('cuda')])
            ground_truth = torch.cat([ground_truth, label.detach()])
        performance(ground_truth, prediction)


if __name__ == "__main__":
    # Dataset List
    D = ['Duolingo', 'MaiMemo']
    Dataset = 'MaiMemo'
    trainLoader, testLoader, valLoader = getLoaders(Dataset)

    if Dataset != 'MaiMemo':
        input_dim = 6
    else:
        input_dim = 5

    hlr_model = HLR(input_dim=input_dim).to('cuda')
    print('HLR')
    test_only(hlr_model, testLoader, Dataset, input_dim)

    wick_model = Wick(input_dim=input_dim).to('cuda')
    print('Wick')
    test_only(wick_model, testLoader, Dataset, input_dim)

    act_model = ACT_R(input_dim=input_dim).to('cuda')
    print('ACT-R')
    test_only(act_model, testLoader, Dataset, input_dim)

    dash_model = DASH(input_dim=input_dim).to('cuda')
    print('DASH')
    test_only(dash_model, testLoader, Dataset, input_dim)

    DSRmodel = DSR()
    print('DSR')
    test_only(DSRmodel, testLoader, Dataset, input_dim)

    SPsyINNC = SPsyINN_C()
    print('SPsyINN-C')
    test_only(SPsyINNC, testLoader, Dataset, input_dim)

    SPsyINNW = SPsyINN_W()
    print('SPsyINN-W')
    test_only(SPsyINNW, testLoader, Dataset, input_dim)

    SPsyINNI = SPsyINN_I()
    print('SPsyINN-I')
    test_only(SPsyINNI, testLoader, Dataset, input_dim)

    PYSR = PySR()
    print('PySR')
    test_only(PYSR, testLoader, Dataset, input_dim)





   ## Training for pam
    # HLRearly_stopping = EarlyStopping(patience=5, verbose=True, function='HLR')
    # hlr_model = HLR(input_dim=input_dim).to('cuda')
    # optimizer = torch.optim.Adam(hlr_model.parameters(), lr=0.01)
    # hlr_criterion = nn.MSELoss()
    # for epoch in range(20):
    #     hlr_model, optimizer = train(hlr_model, trainLoader, hlr_criterion, optimizer, valLoader, HLRearly_stopping, input_dim)
    #     test(hlr_model, testLoader, input_dim)
    # print('HLR')
    # print_model_param_names(hlr_model)
    # print_model_param_values(hlr_model)

    # Wickearly_stopping = EarlyStopping(patience=5, verbose=True, function='Wick')
    # wick_model = Wick(input_dim=input_dim).to('cuda')
    # optimizer = torch.optim.Adam(wick_model.parameters(), lr=0.01)
    # wick_criterion = nn.MSELoss()
    # for epoch in range(20):
    #     wick_model, optimizer = train(wick_model, trainLoader, wick_criterion, optimizer, valLoader, Wickearly_stopping, input_dim)
    #     test(wick_model, testLoader, input_dim)
    # print('Wick')
    # print_model_param_names(wick_model)
    # print_model_param_values(wick_model)

    # ACTearly_stopping = EarlyStopping(patience=5, verbose=True, function='ACT-R')
    # act_model = ACT_R(input_dim=input_dim).to('cuda')
    # optimizer = torch.optim.Adam(act_model.parameters(), lr=0.01)
    # act_criterion = nn.MSELoss()
    # for epoch in range(20):
    #     act_model, optimizer = train(act_model, trainLoader, act_criterion, optimizer, valLoader, ACTearly_stopping, input_dim)
    #     test(act_model, testLoader, input_dim)
    # print('ACT-R')
    # print_model_param_names(act_model)
    # print_model_param_values(act_model)

    # DASHearly_stopping = EarlyStopping(patience=5, verbose=True, function='DASH')
    # dash_model = DASH(input_dim=input_dim).to('cuda')
    # optimizer = torch.optim.Adam(dash_model.parameters(), lr=0.01)
    # dash_criterion = nn.MSELoss()
    # for epoch in range(20):
    #     dash_model, optimizer = train(dash_model, trainLoader, dash_criterion, optimizer, valLoader, DASHearly_stopping, input_dim)
    #     test(dash_model, testLoader, input_dim)
    # print('DASH')
    # print_model_param_names(dash_model)
    # print_model_param_values(dash_model)
