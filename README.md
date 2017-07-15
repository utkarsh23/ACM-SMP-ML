# Digit Classifier with MNIST database
This is a digit classifier that uses MNIST database and implements Artificial Neural Networks (ANN) using numpy, Theano and Lasagne. In layman terms, the machine is trained to recognize a bunch of handwritten digits and then made to predict another bunch of handwritten digits and the final result is the percentage of right predictions.
### Requirements
* python
* numpy
* Theano
* Lasagne
* matplotlib (optional - required for visualizing dataset)
### Visualizing Dataset
To visualize training dataset, execute the following.

`python visualize_training_digits.py`

To visualize training dataset, execute the following.

`python visualize_testing_digits.py`

### Usage
To execute the program, run the following command.

`python digit_classifier.py`

You can also pass an aditional argument to specify the number of iterations.

`python digit_classifier.py [ITERATIONS]`

To learn about usage, run the following command.

`python digit_classifier.py -h`