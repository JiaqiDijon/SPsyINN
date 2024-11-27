import os
import time
import tqdm
import pandas as pd
import model.Constants as C
from pysr.sr import PySRRegressor
from model.data_loader import getLoader, CustomNormalizedDataLoader, MyDataste
from model.GSR_Model import train, test, EarlyStopping,save_epoch



if __name__ == '__main__':

    try:
        e_df = pd.read_csv(C.functionlist[C.Function])
    except pd.errors.ParserError as e:
        print(f"Error reading ALL.csv: {e}")
        raise
    e_df.to_csv('equcation.csv', index=False)

    print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')
    model = PySRRegressor(
        niterations=C.niterations,
        binary_operators=C.binary_operators,
        populations=C.populations,
        population_size=C.population_size,
        ncycles_per_iteration=500,
        equation_file='equcation.csv',
        batching=True,
        batch_size=256,
        precision=64,
        loss="L2DistLoss()",
        maxsize=10,
        warm_start=True,
        maxdepth=C.maxdepth,
        procs=20,
        annealing=True,
        alpha=0.1,
        unary_operators=C.unary_operators,
        model_selection=C.model_selection,
        early_stop_condition="stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
    )
    trainLoaders, testLoaders, valLoaders = getLoader()

    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in tqdm.tqdm(range(C.EPOCH), 'training...'):
        start_time = time.time()
        model = train(model, trainLoaders, early_stopping, valLoaders)
        print(model)
        print(f'epoch :{epoch}, loaded best model')
        ground_truth, prediction = test(model, testLoaders, last=True)
        save_epoch(epoch, ground_truth, prediction)
        model.equations.to_csv(C.Dpath + C.DATASET + '/' + C.Function + str(epoch) + 'function.csv')
        end_time = time.time()
        epoch_time = end_time - start_time
        with open(C.Dpath + C.DATASET + '/GSR-time.txt', 'w+', encoding='utf-8') as f:
            f.write(str(epoch_time))
        if os.path.exists(C.Dpath + C.DATASET + '/' + C.Torch_model_name + '-time.txt'):
            with open(C.Dpath + C.DATASET + '/' + C.Torch_model_name + '-time.txt', 'r', encoding='utf-8') as f:
                DDKTtime = f.read()
                if float(DDKTtime) > epoch_time:
                    time.sleep(float(DDKTtime) - epoch_time)
        else:
            time.sleep(0)

    model = early_stopping.load_checkpoint()
    print("Loaded best model")
    ground_truth, prediction = test(model, testLoaders, last=True)
    model.equations.to_csv(C.Dpath + C.DATASET + '/' + C.Function + 'GSRfunction.csv')
