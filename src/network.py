# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Flatten, Dropout, Lambda
from tensorflow.keras.optimizers import Adam, Adadelta
from datetime import datetime
from src.hyperparams import *

# Creating the Network
def create_network(layer_size=None, predictor=False):

    # If we are iterating over this then use
    # provided layer_size
    if not layer_size:
        layer_size = hidden_layer_size_1
    # Otherwise use the one from hyperparams.py

    # If we are making this network to act as a predictor
    # we want batch size to be single.
    if predictor:
        batch_input_shape = (1, timesteps, n)

    else:
        # Otherwise use the one from hyperparams.py
        batch_input_shape = (batch_size, timesteps, n)

    net = Sequential()
    net.add(LSTM(layer_size, batch_input_shape=batch_input_shape, stateful=True,
        return_sequences=True))
    net.add(LSTM(hidden_layer_size_2, batch_input_shape=batch_input_shape, stateful=True,
        return_sequences=True))
    net.add(LSTM(layer_size, batch_input_shape=batch_input_shape, stateful=True))

    # net.add(Dense(layer_size, batch_input_shape=batch_input_shape))
    net.add(Dense(n))

    optimizer = Adam(learning_rate=learning_rate)
    net.compile(loss=loss_func, optimizer=optimizer)

    return net

def log_config(net, history):
    config = {
        "Running Mean Length" : mean_length,
        "Batch Size" : batch_size,
        "Timesteps" : timesteps,
        "Learning Rate" : learning_rate,
        "Epochs" : epochs,
        "Loss Function": loss_func,
        "Mean Type": mean_type
    }

    configtxt = "\n".join([key + " : " + str(val) for (key, val) in config.items()])

    historytxt = "\n".join([key + " : " + str(val[-1]) for (key, val) in history.history.items()])

    configtxt = "\n" + configtxt + "\n" + historytxt

    with open("logs/run_" + datetime.now().strftime("%y%m%d_%H%M") + ".log", "w") as logfile:
        net.summary(print_fn=lambda x: logfile.write(x + "\n"))
        logfile.write(configtxt)

    net.save("models/run_" + datetime.now().strftime("%y%m%d_%H%M") + ".hdf5")


def trainer(net, x_train, y_train, x_val, y_val, verbose=0):
    """
    Trains the given network with the given data.
    Inputs
    ------
    net: Neural Network/Model instance
    x_train: Training x data
    y_train: Training y data
    x_val: Validation x data
    y_val: Validation y data
    verbose: verbosity level for trainer [0, 1, 2]
    (see keras.models.Sequential.fit for further info)

    Outputs
    -------
    Returns history
    history: network.history object from the network after the training.
    """
    # reset_callback = keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: net.reset_states)
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = net.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          validation_data=(x_val, y_val),
          shuffle=False,
          callbacks=[tensorboard_callback])

    log_config(net, history)

    return history

def predict_from_self(net, x_start, idx_start, idx_end, idx_step):
    """
    Predicts the sequence recursively from the given starting point.

    Inputs
    ------
    net: neural network model to predict with. Should have batch_size=1
    x_start: The x values one batch_size prior to the starting point of
        prediction.
    idx_start: The indexing set corresponding to one batch_size prior to
        the starting point of prediction.
    idx_end: The index at the ending point of the prediction.
    idx_step: The step value for the index. Sets the number of data points
        to predict in the prediction.

    Outputs
    -------
    pred: NumPy array of predictions generated recursively between
        idx_start[-1] and idx_end.
    idx_pred: The index for the predictions made.
    """
    idx_pred = np.arange(idx_start[-1], idx_end, idx_step)
    num_iters = len(idx_pred) + batch_size

    pred = np.zeros(num_iters, dtype=np.float32)
    pred[:batch_size] = x_start[:, 0].reshape(batch_size)

    print("Going to enter the loop now.")
    print("num_iters:", num_iters)
    print("timesteps:", timesteps)
    for i in range(timesteps, num_iters):
        x = pred[i-timesteps:i].reshape(1, timesteps, n)
        y = net.predict(x, batch_size=1)
        if i >= batch_size:
            pred[i] = y

    print("Out of the loop now.")

    return pred[batch_size:], idx_pred
