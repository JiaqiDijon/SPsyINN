import os
import time
import tqdm
import torch
import numpy as np
import torch.optim as optim
import model.Constants as C
from model.data_loader import getLoader, CustomNormalizedDataLoader, MyDataste
from model.TorchModel import DNN, DKT_F, train, test, EarlyStopping, save_epoch

# Set Random Seed
def seed_everything(seed=11):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print('Model: ' + C.Torch_model_name)
    print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')
    if C.Torch_model_name == 'DNN':
        model = DNN(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).to('cuda')
    if C.Torch_model_name == 'DKT-F':
        model = DKT_F(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT).to('cuda')
    seed_everything()

    optimizer = optim.Adam(model.parameters(), lr=C.LR)
    trainLoaders, valLoaders, testLoaders = getLoader()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in tqdm.tqdm(range(C.EPOCH), 'training...'):
        start_time = time.time()
        model, optimizer = train(trainLoaders, valLoaders, model, optimizer, early_stopping, epoch,
                                           weight=True)
        ground_truth, prediction = test(testLoaders, model, last=True)
        save_epoch(epoch, ground_truth, prediction)

        end_time = time.time()
        epoch_time = end_time - start_time

        with open(C.Dpath + C.DATASET + '/' + C.Torch_model_name + '-time.txt', 'w+', encoding='utf-8') as f:
            f.write(str(epoch_time))

        if os.path.exists(C.Dpath + C.DATASET + '/GSR-time.txt'):
            with open(C.Dpath + C.DATASET + '/GSR-time.txt', 'r', encoding='utf-8') as f:
                GPSRtime = f.read()
            if C.Training_model == 'SPsyINN-W':
                if float(GPSRtime) > epoch_time:
                    time.sleep(float(GPSRtime) - epoch_time)
                else:
                    time.sleep(0)
            if C.Training_model == 'SPsyINN-I' and epoch % 2 == 0:
                if float(GPSRtime) > epoch_time:
                    time.sleep(float(GPSRtime) - epoch_time)
                else:
                    time.sleep(0)
            else:
                time.sleep(0)
