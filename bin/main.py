import pandas as pd
import numpy as np
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
data = pd.read_csv(filename, delimiter=";", names=headers)

# Data Preprocessing
x, y = pre.preprocess(data)

if mean_type == "gaussian":
    weights = gaussian(M=mean_length, std=mean_length, sym=True)
elif mean_type == "uniform":
    weights = np.ones(mean_length)/mean_length
else:
    weights = np.ones(mean_length)/mean_length
    
y = pre.running_mean_helper(y, weights)
x_slid, y_slid = pre.sliding_window_main(x, y)
x_train, y_train, x_val, y_val, x_test, y_test = pre.data_splitting_main(x_slid, y_slid)

net = network.create_network()


history = network.trainer(net, x_train, y_train, x_val, y_val, verbose=1)
var_train = np.var(y_train)
var_val = np.var(y_val)
print("Variance in y_train:", var_train)
print("Variance in y_val:", var_val)

plotter.plot_predictions(net, x_train, y_train, x_val, y_val)
plotter.plot_loss_vs_epoch(history, var_train, var_val)


# loss, val_loss = learning_curve()
