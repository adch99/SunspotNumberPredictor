# Sunspot Number Prediction

We use LSTMs to predict the sunspot number data from the observed
data that is available (1818-2019).

## Instructions

In order to use this code to predict sunspots:

1. Clone this repo.
2. Create the directories `img/`, `models/` and `logs`.
3. Run the Jupyter notebook bin/main.ipynb.
4. Change the `%cd` to match the path to the head directory
    on your system.
5. Run all the cells in the notebook.

## Requirements  

* Python 3 (tested with 3.7.7)
* libs: `numpy scipy pandas matplotlib tensorflow_gpu keras_gpu jupyter`

## Notes
* We have used **Adadelta** optimizer. You can use the **Adam** optimizer instead as well. Just change the trainer function in `src/network.py`. Both converge to the same overall minima, Adadelta converges from the top (underfits) while Adam converges from the bottom (overfits). 
