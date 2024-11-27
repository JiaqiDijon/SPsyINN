Dpath = '../dataset/'
# dataset path
datasets = {
    'momo': 'momo/momo_data', # momo -> MaiMemo
    'en_to_de': 'duolingo/en_to_de', # en_to_de -> En2De
    'en_to_es': 'duolingo/en_to_es', # en_to_es -> En2Es
    'all': 'duolingo/all_data' # all -> Duolingo
}
# dataset
DATASET = datasets['all']

# batch size
BATCH_SIZE = 256

# Torch model list
namelist = ['DKT-F', 'DTKT']
Torch_model_name = 'DTKT'

# learning rate
if 'momo' in DATASET:
    LR = 0.01
else:
    LR = 0.001

# epoch
EPOCH = 40

# input dimension
if 'momo' not in DATASET:
    INPUT = 11
else:
    INPUT = 6

# embedding dimension
EMBED = 100
# hidden layer dimension
HIDDEN = 64
# nums of hidden layers
LAYERS = 1
# output dimension
OUTPUT = 1

# GSR Functions parameters for init GSR
functionlist = {'ACTR': 'ACT-R.csv',
                'HLR': 'HLR.csv',
                'Wickelgren': 'Wickelgren.csv',
                'Wixted': 'Wixted.csv',
                'Wozniak': 'Wozniak.csv',
                'ALL': 'ALL.csv',
                'No': 'NO.csv'}
Function = 'ALL'
# GSR parameters
niterations = 40
populations = 40
population_size = 50
maxdepth = 4
binary_operators = ["+", "*", "-", "/", "pow"]
unary_operators = ["exp", "log"]
model_selection = "accuracy" # default "accuracy", "score", "best"  refer to PySR


# Training parameters, asynchronous wait or no wait, or separate training
Training_ = ['SPsyINN-C', 'SPsyINN-W', 'SPsyINN-I', 'O-NN']
Training_model = 'SPsyINN-W'

# Dynamic weighting options: True or False
DAO = True

# dropout rate
DROPOUT = 0.2
