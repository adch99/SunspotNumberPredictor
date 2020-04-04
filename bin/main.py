import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
import src.preprocesser as pre
import src.network as network
import src.plotter as plotter
from src.hyperparams import *

# Getting the data
headers = ["Year",
           "Month",
           "Day",
           "Decimal Date",
           "Daily Total Sunspot Number",
           "Sunspot Number Stddev",
           "No of observations",
           "Definitive/Provisional"
]
filename = "data/SN_d_tot_V2.0.csv"
data = pd.read_csv(filename, delimiter=";", names=headers)[:1000]

# Data Preprocessing
x, y = pre.preprocess(data)

# y_ori = y.copy()


# if mean_type == "gaussian":
#     weights = gaussian(M=mean_length, std=0.1, sym=True)
#     weights /= np.sum(weights) # normalise the weights
#     y = pre.running_mean_helper(y, weights)
# elif mean_type == "uniform":
#     weights = np.ones(mean_length)/mean_length
#     y = pre.running_mean_helper(y, weights)
    # y1 = y.copy()
    # y = pre.running_mean_helper(y, weights)
    # y2 = y.copy()
    # y = pre.running_mean_helper(y, weights)
    # y3 = y.copy()
    # y = pre.running_mean_helper(y, weights)
    # y4 = y.copy()
# else:
#     pass

# debug
# plt.plot(x, y_ori, label="original", alpha=0.5)
# plt.plot(x, y1, label="1st smoothened", alpha=0.6)
# plt.plot(x, y2, label="2nd smoothened", alpha=0.7)
# plt.plot(x, y3, label="3rd smoothened", alpha=0.8)
# plt.plot(x, y4, label="4th smoothened", alpha=0.9)
# plt.legend()
# plt.show()

X = np.array([x, y]).T

x_slid, y_slid = pre.sliding_window_main(X, X)
x_train, y_train, x_val, y_val, x_test, y_test = pre.data_splitting_main(x_slid, y_slid)



net = network.create_network()

history = network.trainer(net, x_train, y_train, x_val, y_val, verbose=2)
var_train = np.var(y_train)
var_val = np.var(y_val)
print("Variance in y_train:", var_train)
print("Variance in y_val:", var_val)

plotter.plot_predictions(net, x_train, y_train, x_val, y_val)
plotter.plot_loss_vs_epoch(history, var_train, var_val)


# loss, val_loss = learning_curve()
