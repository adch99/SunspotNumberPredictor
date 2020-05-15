import numpy as np

# Hyper-Parameters
n = 1 # Number of features
mean_length = 30 # redundant unless mean_type is not "none"
batch_size = 32
timesteps = 4
learning_rate = 0.01
epochs = 100
hidden_layer_size_1 = 4
hidden_layer_size_2 = 10 # redundant
loss_func = "mse"
data_split_ratio = np.array([0.6, 0.2, 0.2])
mean_type = "uniform"
