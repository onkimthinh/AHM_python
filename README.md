# AHM_python
Assisted History Matching (AHM) using machine learning workflow

The idea of this project is to fit the production data using reservoir and operation parameters as model inputs. 
The Artificial Neural Network (ANN) models are used as the prediction model.
The optimization is stochastic optimization using Particle Swarm Optimization (PSO) and Differential Evolution (DE)
Stochastic optimization algorithms search randomly through the input space and yield an optimal set of input parameters that minimizes the history matching error.


# ANN_training.py: train multiple fully-connected neural networks and select the best model architectures for later steps. The selected ANN model will be used as the feedforward model for stochastic optimization.
# ANN_PSO.py: use PSO to conduct stochastic optimization, search for the best set of input parameters to minimize history matching error
# ANN_DE.py: similar to ANN_PSO.py, but using DE instead of PSO

# Steps to run the history matching framework:
1. Run "ANN_training.py" to train the Neural Networks (ANN) and select the best model architecture

2. Run "ANN_PSO.py" or "ANN_DE.py" for stochastic optimization
