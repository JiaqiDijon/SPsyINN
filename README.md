# SPsyINN：Combining Denoised Neural Network and Genetic Symbolic Regression for Memory Behavior Modeling via Dynamic Asynchronous Optimization

![avator](Images/SPsyINN.png)

**SPsyINN** is a novel neural network method that integrates data and knowledge, significantly enhancing the modeling of learners’ memory processes through knowledge injection. We demonstrated that **dynamic asynchronous optimization** is effective in resolving interaction challenges between neural networks and symbolic regression. In practical applications, SPsyINN uncovered memory equations consistent with classical theories while revealing the dual influence of time intervals and learners’ historical behaviors, offering valuable insights for memory modeling.

In summary, we have, for the first time, integrated machine learning with cognitive science, providing a psychologically interpretable dynamic asynchronous training model that opens new possibilities for personalized education and the discovery of memory laws.

---

## Project Execution

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

## Dataset

The `.zip` files in the `dataset` directory contain the data required for training. These files need to be extracted into the `dataset` directory.
Similarly, to facilitate the symbolic regression process, we have provided corresponding `.csv` files. These files are stored in the `.zip` archive within the `dataset(csv)` directory and must be extracted to access the respective files.


## For Functions performance

We have configured corresponding methods to evaluate equation performance in the `Test for functions` directory for your convenience.



