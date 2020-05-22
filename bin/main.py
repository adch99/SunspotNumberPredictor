# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

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
data = pd.read_csv(filename, delimiter=";", names=headers)[:6000]

# Data Preprocessing
dates, spots, inverter = pre.preprocess(data)

X = spots
index = dates
x_slid, y_slid, idx_slid = pre.sliding_window_main(X, X, index)
x_train, y_train, idx_train, x_val, y_val, idx_val, x_test, y_test, idx_test = pre.data_splitting_main(x_slid, y_slid, idx_slid)


net = network.create_network()

history = network.trainer(net, x_train, y_train, x_val, y_val, verbose=1)
var_train = np.var(y_train)
var_val = np.var(y_val)
print("Variance in y_train:", var_train)
print("Variance in y_val:", var_val)

plotter.plot_predictions(net, x_train, y_train, idx_train, x_val, y_val, idx_val)
plotter.plot_loss_vs_epoch(history, var_train, var_val)

plotter.plot_weights(net)
plt.close()
predictor = network.create_network(predictor=True)
predictor.set_weights(net.get_weights())

args = (
    predictor,
    x_train,
    y_train,
    idx_train
)
predictor.reset_states()
plotter.plot_recursive_predictions(*args)
plt.close()
