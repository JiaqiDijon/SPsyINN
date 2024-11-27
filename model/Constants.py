Dpath = '../dataset/'
# dataset path
datasets = {
    'MaiMemo': 'MaiMemo/',  # MaiMemo
    'en_to_de': 'En2De/',  # en_to_de -> En2De
    'en_to_es': 'En2Es/',  # en_to_es -> En2Es
    'Duolingo': 'Duolingo/'  # Duolingo
}
# dataset
DATASET = datasets['Duolingo']

# Torch model list
namelist = ['DKT-F', 'DNN']
Torch_model_name = 'DNN'

# input dimension
if 'MaiMemo' not in DATASET:
    INPUT = 6
else:
    INPUT = 5
    
# learning rate
if 'MaiMemo' in DATASET:
    LR = 0.01
else:
    LR = 0.001

# batch size
BATCH_SIZE = 256

# epoch
EPOCH = 40

# embedding dimension
EMBED = 100

# hidden layer dimension
HIDDEN = 64

# nums of hidden layers
LAYERS = 1

# output dimension
OUTPUT = 1

# dropout rate
DROPOUT = 0.2

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
niterations = 40 # Number of iterations for the genetic algorithm.
populations = 40 # Number of populations
population_size = 50 # Number of populations_size
maxdepth = 4 # Operator nesting depth.
binary_operators = ["+", "*", "-", "/", "pow"] # Operator set.
unary_operators = ["exp", "log"] # Additional operator set.
model_selection = "accuracy" # default "accuracy", "score", "best"  refer to PySR

# Training parameters, asynchronous wait or no wait, or separate training
Training_ = ['SPsyINN-C', 'SPsyINN-W', 'SPsyINN-I', 'O-NN'] 
Training_model = 'SPsyINN-W'

# Dynamic weighting options: True or False
DAO = True
