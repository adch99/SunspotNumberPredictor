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
dates, spots, inverter = pre.preprocess(data)

if mean_type == "gaussian":
    weights = gaussian(M=mean_length, std=0.1, sym=True)
    weights /= np.sum(weights) # normalise the weights
    spots = pre.running_mean_helper(spots, weights)
elif mean_type == "uniform":
    weights = np.ones(mean_length)/mean_length
    spots = pre.running_mean_helper(spots, weights)
else:
    pass

X = np.diff(spots)
index = dates[1:]
for predict_ahead in range(0, 20):
    x_slid, y_slid, idx_slid = pre.sliding_window_main(X, X, index=index, predict_ahead=predict_ahead)
    x_train, y_train, idx_train, x_val, y_val, idx_val, x_test, y_test, idx_test = pre.data_splitting_main(x_slid, y_slid, idx_slid, output=False)

    net = network.create_network()

    history = network.trainer(net, x_train, y_train, x_val, y_val, verbose=0)
    var_train = np.var(y_train)
    var_val = np.var(y_val)
    #print("Variance in y_train:", var_train)
    #print("Variance in y_val:", var_val)
    loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]
    print("predict_ahead: %d - normed loss: %.4f - normed val_loss: %.4f" % (predict_ahead, loss/var_train, val_loss/var_val))
    print()

    plotter.plot_predictions(net, x_train, y_train, idx_train,
        x_val, y_val, idx_val)
    plt.close()
    plotter.plot_loss_vs_epoch(history, var_train, var_val)
    plt.close()
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
