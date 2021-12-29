"""Model definitions."""

import numpy as np
from typing import Tuple
import torch
from torch.nn import Module, Linear, ReLU, Softmax, Sequential, CrossEntropyLoss
from torch.nn.modules import linear
from dataset import X, y


def one_hot_encoding(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one hot encoding.

    Example: y=[1, 2], num_classes=3 --> [[0, 1, 0], [0, 0, 1]]

    Args:
        y: Input labels as integers with shape (num_datapoints)
        num_classes: Number of possible classes

    Returns:
        One-hot encoded labels with shape (num_datapoints, num_classes)
    """
    encoded = np.zeros(y.shape + (num_classes,))
    encoded[np.arange(len(y)), y] = 1
    return encoded

def create_2unit_net() -> Module:
    """Create a two-layer MLP (1 hidden layer, 1 output layer) with 2 hidden units as described in the exercise.

    Returns:
        2-layer MLP module with 2 hidden units.
    """
    # START TODO #################
    # Define the model here
    linear_units = 2
    model = Sequential(Linear(2, linear_units),
                        ReLU(),
                        Linear(linear_units, 2)    
                    )

    # END TODO ##################
    params = model.state_dict()

    # Now we assign the model weights since we still did not learn backpropagation
    params['0.weight'] = torch.Tensor(np.array([[3.21, 3.21], [-2.34, -2.34]]))
    params['0.bias'] = torch.Tensor(np.array([-3.21, 2.34]))
    params['2.weight'] = torch.Tensor(np.array([[3.19, 4.64], [-2.68, -3.44]]))
    params['2.bias'] = torch.Tensor(np.array([-4.08, 4.42]))
    model.load_state_dict(params)


    return model


def create_3unit_net() -> Module:
    """Create a two-layer MLP (1 hidden layer, 1 output layer) with 3 hidden units as described in the exercise.

    Returns:
        2-layer MLP module with 3 hidden units.
    """
    # START TODO #################
    # Define the model here
    linear_units = 3
    model = Sequential(
                    Linear(2, linear_units),
                    ReLU(),
                    Linear(linear_units, 2)
    )
    # END TODO ##################

    params = model.state_dict()
    # START TODO #################
    # change the model weights

    params['0.weight'] = torch.Tensor(np.array([[3.21, 3.21], [-2.34, -2.34], [0,0]]))
    params['0.bias'] = torch.Tensor(np.array([-3.21, 2.34, 0]))
    params['2.weight'] = torch.Tensor(np.array([[3.19, 4.64, 0], [-2.68, -3.44, 0]]))
    params['2.bias'] = torch.Tensor(np.array([-4.08, 4.42]))
    model.load_state_dict(params)

    # END TODO ##################
    model.load_state_dict(params)

    return model


def run_model_on_xor(model: Module, verbose: bool = True) -> Tuple[np.ndarray, np.float]:
    """Run the XOR dataset through the model and compute the loss.

    Args:
        model: MLP to use for prediction
        verbose: Whether to print the outputs.

    Returns:
        Tuple containing:
            Class predictions after softmax with shape (batch_size, num_classes)
            Cross-Entropy loss given the model outputs and the true labels

    """
    # Here we test if our prediction works. We first get the so-called "logits" (the MLP output before the softmax),
    # then run them through the softmax function. We have to transform the prediction into one-hot format,
    # and finally we can check whether our MLP predicts the correct values.

    # START TODO #################
    # propagate the input data (stored in the imported variable X) through the model.
    prediction = model(torch.Tensor(X))

    # END TODO ##################

    if verbose:
        print("Raw prediction logits:")
        print(prediction.detach().cpu().numpy())
        print()
    softmax_function = Softmax(dim=-1)
    pred_softmax = softmax_function(prediction).cpu().detach().numpy()
    if verbose:
        print("Prediction after softmax:")
        print(pred_softmax)
        print()

    # START TODO #################
    # Use the one_hot_encoding function on the labels to convert them to
    # one-hot encoding. The labels are stored in the imported variable y.
    Y_onehot = one_hot_encoding(y, 2)
    # END TODO ##################

    if verbose:
        print("True labels, one-hot encoded:")
        print(Y_onehot)
        print()

    # Pass prediction and ground truth to the generalized Cross-Entropy Loss.
    # Since the loss has a softmax already implemented inside of it, you need to pass the raw logits of the
    # prediction. The loss expects one-hot encoded labels of shape (batchsize, num_classes)
    loss_fn = CrossEntropyLoss()

    # given the true labels Y and the predictions,
    # compute the cross entropy loss defined above
    loss = loss_fn(prediction, torch.LongTensor(y)).cpu().detach().numpy()

    if verbose:
        print("Loss:", loss)

    # return predictions and loss for testing
    return pred_softmax, loss


def run_test_model(model: Module) -> None:
    """Helper function to test if the model predicts the correct classes.

    Args:
        model: Module to predict the classes.

    Returns:
        None
    """
    pred_softmax, loss = run_model_on_xor(model, verbose=False)
    Y_onehot = one_hot_encoding(y, 2)
    np.testing.assert_allclose(
        pred_softmax, Y_onehot, atol=1e-3,
        err_msg=f"The model predicts the wrong classes. Ground-truth: {Y_onehot}, predictions: {pred_softmax}")
    assert np.abs(loss) < 1e-3, f"Loss is too high: {loss}"
