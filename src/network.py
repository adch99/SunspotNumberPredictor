# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Flatten, Dropout
from keras.optimizers import Adam, Adadelta
# import tensorflow as tf
from datetime import datetime
from src.hyperparams import *

# Creating the Network
def create_network():
    # weights = np.random.normal(hidden_layer_size_1*n, mu=0, sigma=1)
    batch_input_shape = (batch_size, timesteps, n)
    net = Sequential()
    # net.add(LSTM(hidden_layer_size_1, batch_input_shape=batch_input_shape, stateful=True, kernel_initializer='RandomNormal', bias_initializer='ones'))
    net.add(LSTM(hidden_layer_size_1, batch_input_shape=batch_input_shape))
    # net.add(Activation("sigmoid"))
    net.add(Dense(n))
    # net.add(Activation("sigmoid"))

    optimizer = Adadelta()

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


def trainer(net, x_train, y_train, x_val, y_val, verbose=0):
    #print('Training')
    callback = keras.callbacks.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: net.reset_states)
    history = net.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          validation_data=(x_val, y_val),
          shuffle=False,
          callbacks=[callback])


    log_config(net, history)

    return history
