import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import tensorflow as tf
from src.hyperparams import *

# Creating the Network
def create_network():
    input_shape = (timesteps, n)
    net = Sequential()
    net.add(Dense(hidden_layer_size_1, input_shape=input_shape, batch_size=batch_size, activation="linear"))
    net.add(Dense(n))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    net.compile(loss=loss_func, optimizer=optimizer)

    return net

def log_config():
    config = {
        "Running Mean Length" : mean_length,
        "Batch Size" : batch_size,
        "Timesteps" : timesteps,
        "Learning Rate" : learning_rate,
        "Epochs" : epochs,
        "Loss Function": loss_func
    }

    layers = "Hidden Layer : " + str(hidden_layer_size_1)
    configtxt = "\n".join([key + " : " + str(val) for (key, val) in config.items()]) + "\n" + layers

    print(configtxt)
    with open("logs/run_" + datetime.now().strftime("%y%m%d_%H%M") + ".log") as logfile:
        logfile.write(configtxt)


def trainer(net, x_train, y_train, x_val, y_val, verbose=0):
    #print('Training')
    history = net.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          validation_data=(x_val, y_val),
          shuffle=False)

    return history