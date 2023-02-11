import datasets

import numpy as np
import os
import time
import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

if torch.cuda.is_available():
  use_cuda = True
else:
  use_cuda = False

###############################################################################
# For Training
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path


###############################################################################
# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


    def get_accuracy(net, loader, label_names):
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if use_cuda and torch.cuda.is_available():
          inputs= inputs.cuda()
          labels = labels.cuda()
          net = net.cuda()
        outputs = net(inputs)
        labels.float()
        classification_report(
            labels,
            outputs,
            output_dict=False,
            target_names=label_names
        )
    return


def get_confusion_matrix(net: nn.Module, loader: torch.utils.data.DataLoader, label: str):
    """
    Returns the relevant confusion matrix corresponding to the input model and label
    :param net: the model being tested
    :param loader: the test data
    :param label: the label being predicted

    NOTE: the label must be in the list ["sentiment", "respect", "insult",
            "humiliate", "status", "dehumanize", "violence", "genocide"
            "attack_defend"]
    """
    index = {"sentiment": 0, "respect": 1, "insult": 2, "humiliate": 3, "status": 4,
             "dehumanize": 5, "violence": 6, "genocide": 7, "attack_defend": 8}

    actual = []
    predicted = []
    relevant_index = index[label]

    # get actual stuff
    for _, data in enumerate(loader, 0):
        inputs, labels = data
        curr_tensor = torch.FloatTensor(inputs)

        actual_value = labels[relevant_index]  # how do I get the actual stuff
        predicted_value = (net.forward(curr_tensor))[relevant_index]

        # using 0.5 cutoff

        if actual_value > 0.5:
            actual.append(1)
        else:
            actual.append(0)

        if predicted_value > 0.5:
            predicted.append(1)
        else:
            predicted.append(0)

    # make confusion matrix
    for i in range(len(actual)):
        actual[i] = pd.Series(actual[i], name=('Actual ' + label))
        predicted[i] = pd.Series(predicted[i], name=('Predicted ' + label))
        print(pd.crosstab(actual[i], predicted[i]))


###############################################################################
def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if use_cuda and torch.cuda.is_available():
          inputs= inputs.cuda()
          labels = labels.cuda()
          net = net.cuda()
        outputs = net(inputs)
        loss_func = nn.BCELoss()
        loss = loss_func(outputs, labels.float())
        loss.backward()
    return_loss = float(loss)
    return return_loss

def train_net(net, train_loader, val_loader, batch_size=150, learning_rate=0.005, num_epochs=6):
    # Define the Loss function and optimizer
    # Optimizer will be SGD with Momentum.
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        i=0
        for text, labels in iter(train_loader):
            if use_cuda and torch.cuda.is_available():
              text = text.cuda()
              labels = labels.cuda()
              net = net.cuda()
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(text)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            i=i+1
        train_loss[epoch] = float(loss)
        # train_acc[epoch] = get_accuracy(net, train_loader, label_names)
        train_acc[epoch] = 0
        # val_acc[epoch] = get_accuracy(net, valid_loader, label_names)
        val_acc[epoch] = 0
        val_loss[epoch] = evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train acc: {}, Train loss: {} | "+
               "Validation acc: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_acc[epoch],
                   train_loss[epoch],
                   val_acc[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch+1)
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_err.csv".format(model_path), train_acc)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_acc)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
