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
data = pd.read_csv(filename, delimiter=";", names=headers)[:200]

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

predictor = network.create_network(predictor=True)
predictor.set_weights(net.get_weights())

idx_step = np.average(np.diff(idx_slid))
x_start = x_val[:batch_size, :, :]
idx_start = idx_val[:batch_size]
idx_end = idx_val[-1]
args = (
    predictor,
    x_start,
    idx_start,
    idx_end,
    idx_step
)
predictor.reset_states()
# net.reset_states()
npred = predictor.predict(x_val, batch_size=1)
rpred, idx_rpred = network.predict_from_self(*args)
loss_func = keras.losses.MeanSquaredError()
print("Loss: %.4f" % loss_func(y_val, rpred))

plt.figure(figsize=(10,8))
plt.plot(idx_val, y_val, label="Actual Value", marker="+")
plt.plot(idx_rpred, rpred, label="Recursive Prediction", marker="o")
plt.plot(idx_val, npred, label="Normal Prediction", marker="x")

plt.xlabel("Date")
plt.ylabel("Normalised Sunspot Numbers")
plt.legend()
