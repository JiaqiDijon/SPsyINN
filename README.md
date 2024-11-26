# SPsyINN：Combining Denoised Neural Network and Genetic Symbolic Regression for Memory Behavior Modeling via Dynamic Asynchronous Optimization

![avator](Images/SPsyINN.png)

**SPsyINN** is a novel neural network method that integrates data and knowledge, significantly enhancing the modeling of learners’ memory processes through knowledge injection. We demonstrated that **dynamic asynchronous optimization** is effective in resolving interaction challenges between neural networks and symbolic regression. In practical applications, SPsyINN uncovered memory equations consistent with classical theories while revealing the dual influence of time intervals and learners’ historical behaviors, offering valuable insights for memory modeling.

In summary, we have, for the first time, integrated machine learning with cognitive science, providing a psychologically interpretable dynamic asynchronous training model that opens new possibilities for personalized education and the discovery of memory laws.

---

# Project Implementation

Our project is straightforward to implement. You can execute the `./trainer/run.py` script to run the project.

## Configuration Details

The configuration details are located in the `model/Constants.py` file. In this file, you can modify various parameters:

- **Training dataset**
- **LSTM configuration** in the Denoised Neural Network (DNN)
- **Training strategies**
- Enabling or disabling **Dynamic Asynchronous Optimization (DAO)**
- Initialization equations for **Genetic Symbolic Regression (GSR)**

Additionally, settings like the number of iterations and population size for GSR can be adjusted in the model initialization section of `trainer/GPSR_train.py`.

## Training Details

- **DNN training** specifics are located in `trainer/TorchModel_train.py`.
- **GSR training** specifics can be found in `trainer/GPSR_train.py`.

## For Functions performance

We have configured corresponding methods to evaluate equation performance in the `Test for functions` directory for your convenience.


We will further enhance our codebase to make it easier for you to deploy.
