import numpy as np

# Hyper-Parameters
n = 1 # Number of features
mean_length = 10
batch_size = 32
timesteps = 10
learning_rate = 0.001
epochs = 20
hidden_layer_size_1 = 64
hidden_layer_size_2 = 64
loss_func = "mse"
data_split_ratio = np.array([0.6, 0.2, 0.2])
mean_type = "uniform"