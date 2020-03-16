from datetime import datetime
import matplotlib.pyplot as plt
from src.hyperparams import *

def plot_predictions(net, x_train, y_train, x_val, y_val):
    # Plot the prediction on the training and validation set
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 30))
    pred1 = net.predict(x_val, batch_size=batch_size)
    ax1.plot(y_val, label="Actual Data")
    ax1.plot(pred1, label="Prediction")
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sunspot Numbers")
    ax1.legend()
    ax1.set_title("Predicted vs Actual Validation Data")

    pred2 = net.predict(x_train, batch_size=batch_size)
    ax2.plot(y_train, label="Actual Data")
    ax2.plot(pred2, label="Prediction")
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Sunspot Numbers")
    ax2.legend()
    ax2.set_title("Predicted vs Actual Training Data")

    fig.savefig("img/" + datetime.now().strftime("%y%m%d_%H%M") + "_predicted_vs_actual_data.png", format="png")

def plot_loss_vs_epoch(history, var_train, var_val):
    # Plot training & validation loss values
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.plot(history.history['loss']/var_train, marker="o")
    plt.plot(history.history['val_loss']/var_val, marker="o")
    plt.title('Model Loss')
    plt.ylabel('Loss (Normalised to variance of dataset)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    # plt.ylim(bottom=0)
    plotfilename = "img/" + datetime.now().strftime("%y%m%d_%H%M") + "_model_loss.png"
    plt.savefig(plotfilename)
    plt.show()

def learning_curve():
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