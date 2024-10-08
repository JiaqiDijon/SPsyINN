Dpath = '../dataset/'
# dataset path
datasets = {
    'momo': 'momo/momo_data',
    'en_to_de': 'duolingo/en_to_de',
    'en_to_es': 'duolingo/en_to_es',
    'all': 'duolingo/all_data'
}
# dataset
DATASET = datasets['all']

# seq size
if 'momo' in DATASET:
    MAX_STEP = 5
else:
    MAX_STEP = 16

# batch size
BATCH_SIZE = 256

# Torch model list
namelist = ['DKT', 'DTKT', 'FIFKT']
Torch_model_name = 'DTKT'

# learning rate
if 'momo' in DATASET:
    LR = 0.01
else:
    LR = 0.001

# epoch
EPOCH = 40

# input dimension
if Torch_model_name == 'FIFKT' and 'momo' not in DATASET:
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

# GPSR Functions parameters
functionlist = {'ACTR': 'ACT-R.csv',
                'HLR': 'HLR.csv',
                'Wickelgren': 'Wickelgren.csv',
                'Wixted': 'Wixted.csv',
                'Wozniak': 'Wozniak.csv',
                'ALL': 'ALL.csv',
                'No': 'NO.csv'}
Function = 'ALL'

# Training parameters, asynchronous wait or no wait, or separate training
Training_ = ['Asy-00', 'Asy-11', 'Asy-21', 'O-NN']
Training_model = 'Asy-11'

# Dynamic weighting options: True or False
DyOp = True

# dropout rate
DROPOUT = 0.2
