# SPsyINN：Combining Denoised Neural Network and Genetic Symbolic Regression for Memory Behavior Modeling via Dynamic Asynchronous Optimization

![avator](Images/SPsyINN.png)

**SPsyINN** is a novel neural network method that integrates data and knowledge, significantly enhancing the modeling of learners’ memory processes through knowledge injection. We demonstrated that **dynamic asynchronous optimization** is effective in resolving interaction challenges between neural networks and symbolic regression. In practical applications, SPsyINN uncovered memory equations consistent with classical theories while revealing the dual influence of time intervals and learners’ historical behaviors, offering valuable insights for memory modeling.

In summary, we have, for the first time, integrated machine learning with cognitive science, providing a psychologically interpretable dynamic asynchronous training model that opens new possibilities for personalized education and the discovery of memory laws.

---
## Dataset

The `.zip` files in the `dataset` directory contain the data required for training. These files need to be extracted into the `dataset` directory.
- To facilitate the validation of symbolic regression, we have organized the corresponding data into CSV files stored in the `dataset(csv)` directory. You need to extract the compressed files in this directory.
- The data includes (clearly labeled in the CSV files):
  - $\delta_{1:6}$: Represents user memory features. In the Duolingo, En2De and En2Es dataset, $\delta_{1:6}$ is intact. However, in the MaiMemo dataset, $\delta_3$ was not available, so we retained only $\delta_1, \delta_2, \delta_4, \delta_5, \delta_6$.
  - $Recall$: Represents memory states.

## Project Execution

## Quick Start
### Environment Configuration
```bash
conda create -n spsyinn python=3.8
conda activate spsyinn
pip install -r requirements.txt




Our project is straightforward to implement. You can execute the `./trainer/run.py` script to run the project. 

**Note:** You need to extract the `.zip` files in the `dataset` directory before running the script.

## Configuration Details
The configuration details are located in the `model/Constants.py` file. In this file, you can modify various parameters:

- **Training dataset**
- **LSTM configuration** in the Denoised Neural Network (DNN)
- **Training strategies**
- Enabling or disabling **Dynamic Asynchronous Optimization (DAO)**
- Initialization equations for **Genetic Symbolic Regression (GSR)**
- settings like the number of iterations and population size for GSR

## Training Details

- **DNN training** specifics are located in `trainer/TorchModel_train.py`.
- **GSR training** specifics can be found in `trainer/GPSR_train.py`.


@inproceedings{10.1145/3711896.3736886,
author = {Sun, Jianwen and Chen, Qirong and Huang, Zhenya and Hu, Zhihai and Liang, Ruxia and Shen, Xiaoxuan},
title = {Combining Denoised Neural Network and Genetic Symbolic Regression for Memory Behavior Modeling via Dynamic Asynchronous Optimization},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3736886},
doi = {10.1145/3711896.3736886},
abstract = {Memory behavior modeling is a key topic in cognitive psychology and education. Traditional approaches use experimental data to build memory equations, but these models often lack precision and are debated in form. Recently, data-driven methods have improved predictive accuracy but struggle with interpretability, limiting cognitive insights. Although knowledge-informed neural networks have succeeded in fields like physics, their use in behavior modeling is still limited. This paper proposes a Self-evolving Psychology-informed Neural Network (SPsyINN), which leverages classical memory equations as knowledge modules to constrain neural network training. To address challenges such as the difficulty in quantifying descriptors and the limited interpretability of classical memory equations, a genetic symbolic regression algorithm is introduced to conduct evolutionary searches for more optimal expressions based on classical memory equations, enabling the mutual progress of the knowledge module and the neural network module. Specifically, the proposed approach combines genetic symbolic regression and neural networks in a parallel training framework, with a dynamic joint optimization loss function ensuring effective knowledge alignment between the two modules. Then, for addressing the training efficiency differences arising from the distinct optimization methods and computational hardware requirements of genetic algorithms and neural networks, an asynchronous interaction mechanism mediated by proxy data is developed to facilitate effective communication between modules and improve optimization efficiency. Finally, a denoising module is integrated into the neural network to enhance robustness against data noise and improve generalization performance. Experimental results on five large-scale real-world memory behavior demonstrate that SPsyINN outperforms state-of-the-art methods in predictive accuracy. Ablation studies confirm the model's co-evolution capability, improving accuracy while discovering more interpretable memory equations, showing its potential for psychological research. Our code is released at: https://github.com/JiaqiDijon/SPsyINN},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {2735–2746},
numpages = {12},
keywords = {cognitive psychology, genetic symbolic regression, knowledge-informed neural networks, memory behavior modeling},
location = {Toronto ON, Canada},
series = {KDD '25}
}
