import sys
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28)
    return data / np.float32(256)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def mlp_architecture(input_var=None):
    print("    Input layer:\t28x28=784 units")
    print("    Hidden layer:\t1000 units")
    print("    Output layer:\t10 units\n")
    inp_layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    hidden_layer = lasagne.layers.DenseLayer(inp_layer, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
    output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
    return output_layer

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_iterations=20):
    print("[\u2022] Loading data...\n")
    train_images = load_mnist_images('train-images.idx3-ubyte')
    train_labels = load_mnist_labels('train-labels.idx1-ubyte')
    X_train = train_images[:-10000]
    X_val = train_images[-10000:]
    y_train = train_labels[:-10000]
    y_val = train_labels[-10000:]
    X_test = load_mnist_images('t10k-images.idx3-ubyte')
    y_test = load_mnist_labels('t10k-labels.idx1-ubyte')

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("[\u2022] Building network and compiling functions...\n")
    network = mlp_architecture(input_var)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    print("    Update method:\tStochastic Gradient Descent (SGD)")
    print("    Learning rate:\t1.0\n")    
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=1.0)

    test_prediction = lasagne.layers.get_output(network)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("[\u2022] Starting training...\n")
    for iteration in range(num_iterations):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 10000, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("    Iteration {} of {} took {:.3f}s".format(iteration + 1, num_iterations, time.time() - start_time))
        print("      training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("      validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("      validation accuracy:\t{:.2f} %\n".format(val_acc / val_batches * 100))

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 10000, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("\n[\u2022] Final results:\n")
    print("    Test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("    Test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("-----------")
        print("Usage:\npython %s [ITERATIONS]\n" % sys.argv[0])
        print("ITERATIONS: number of training iterations to perform (default: 20)")
        print("-----------")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_iterations'] = int(sys.argv[1])
        main_start = time.time()
        main(**kwargs)
        print("    Total time elapsed:\t\t{:.3f}s".format(time.time() - main_start))