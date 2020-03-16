# Sunspot Number Prediction

We will be using LSTMs and Neural Networks to predict sunspot numbers
in the future.

* We first smoothen the curve by taking a running mean
    - gaussian-weighted mean
    - uniform weighted mean

* Then we mean-normalise the data to between -1 and 1.