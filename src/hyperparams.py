import numpy as np

# Hyper-Parameters
n = 2 # Number of features
mean_length = 30
batch_size = 1
timesteps = 10
learning_rate = 1
epochs = 10
hidden_layer_size_1 = 4
hidden_layer_size_2 = 4
loss_func = "mse"
data_split_ratio = np.array([0.6, 0.2, 0.2])
mean_type = "none"