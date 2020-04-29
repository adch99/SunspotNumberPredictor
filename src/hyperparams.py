import numpy as np

# Hyper-Parameters
n = 1 # Number of features
mean_length = 30
batch_size = 1
timesteps = 2
learning_rate = 0.05
epochs = 100
hidden_layer_size_1 = 100
hidden_layer_size_2 = 10
loss_func = "mse"
data_split_ratio = np.array([0.6, 0.2, 0.2])
mean_type = "none"
