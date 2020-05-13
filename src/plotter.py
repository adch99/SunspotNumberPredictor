from datetime import datetime
import matplotlib.pyplot as plt
from src.hyperparams import *

def plot_predictions(net, x_train, y_train, idx_train, x_val, y_val, idx_val):
    """
    Plot the prediction on the training and validation set.
    Inputs
    -------
    net: The neural network/model instance
    x_train: Training x data
    y_train: Training y data
    idx_train: Indexing set for the training data (usually the date)
    x_val: Validation x data
    y_val: Validation y data
    idx_val: Indexing set for the validation data (usually the date)

    Outputs
    -------
    Returns None
    Saves plot generated to
    img/[current date & time]_predicted_vs_actual_data.png
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 30))
    pred1 = net.predict(x_val, batch_size=batch_size)
    print("pred1.shape:", pred1.shape)
    ax1.plot(idx_val, y_val, label="Actual Data", marker="+")
    ax1.plot(idx_val, pred1, label="Prediction", marker="o")
    # ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sunspot Numbers")
    ax1.legend()
    ax1.set_title("Predicted vs Actual Validation Data")

    pred2 = net.predict(x_train, batch_size=batch_size)
    print("pred2.shape:", pred2.shape)
    ax2.plot(idx_train, y_train, label="Actual Data", marker="+")
    ax2.plot(idx_train, pred2, label="Prediction", marker="o")
    # ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Sunspot Numbers")
    ax2.legend()
    ax2.set_title("Predicted vs Actual Training Data")

    plt.tight_layout()

    filename = "img/"
    filename += datetime.now().strftime("%y%m%d_%H%M")
    filename += "_predicted_vs_actual_data.png"
    fig.savefig(filename, format="png")

def plot_loss_vs_epoch(history, var_train, var_val, show=True):
    """
    Plot training & validation loss values.
    Inputs
    ------
    history: The network.history object after training the network
    var_train: Variance of the training y data
    var_val: Variance of the validation y data

    Outputs
    -------
    Returns None
    Saves plot generated to img/[current date & time]_model_loss.png
    """
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.plot(history.history['loss']/var_train, marker="o")
    plt.plot(history.history['val_loss']/var_val, marker="o")
    plt.title('Model Loss')
    plt.ylabel('Loss (Normalised to variance of dataset)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    # plt.ylim(bottom=0)
    filename = "img/"
    filename += datetime.now().strftime("%y%m%d_%H%M")
    filename += "_model_loss.png"
    plt.savefig(filename)

    if show:
        plt.show()

def learning_curve():
    """
    Generates a 'learning curve' i.e. the analysis for each data size 0 to full.
    Obsolete function.
    """
    loss = []
    val_loss = []
    data_size = []

    x_slid, y_slid = sliding_window_main(x, y)
    x_train, y_train, x_val, y_val, x_test, y_test = data_splitting_main(x_slid, y_slid)
    m_tot = x_train.shape[0]

    batch_step = 50
    try:
        for m in range(batch_size, m_tot, batch_step*batch_size):
            print("Training: ", m)
            net = create_network()
            history = trainer(net, x_train[:m], y_train[:m], x_val, y_val)
            loss.append(history.history["loss"][-1])
            val_loss.append(history.history["val_loss"][-1])
            data_size.append(m)

        print("Loss:", loss[-1])
        print()

    finally:
        plt.plot(data_size, loss, label="Loss", marker="o")
        plt.plot(data_size, val_loss, label="Val Loss", marker="o")
        plt.xlabel("m")
        plt.ylabel("Losses")
        plt.title("Model Loss")
        plt.legend()
        plt.savefig("img/" + datetime.now().strftime("%y%m%d_%H%M") + "_learning_curve.png")
        plt.show()

    return loss, val_loss
