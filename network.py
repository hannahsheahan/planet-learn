
"""
Author: Hannah Sheahan, sheahan.hannah@gmail.com
Date: 19/11/2020
Notes: N/A
Issues: N/A
"""
import numpy as np
import sys
import constants as const

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import random
from datetime import datetime
from IPython import embed
import svd_plots
from scipy.stats import zscore


class CreateDataset(Dataset):
    """A class to hold a dataset."""

    def __init__(self, dataset, transform=None):
        """
        Args:
            datafile (string): name of numpy datafile
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.index = dataset['index']
        self.label = dataset['label']
        self.input_features = dataset['input_features']
        self.data = {'index':self.index, 'label':self.label,  'input_features':self.input_features}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # for retrieving either a single sample of data, or a subset of data
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'index':self.index[idx], 'label':self.label[idx].flatten(), 'input_features':self.input_features[idx]}
        return sample


def batch_to_torch(originalimages):
    """Convert the input batch to a torch tensor"""
    #originalimages = originalimages.unsqueeze(1)   # change dim for the convnet
    originalimages = originalimages.type(torch.FloatTensor)  # convert torch tensor data type
    return originalimages


def train(args, model, device, train_loader, optimizer, criterion, epoch, printOutput=True):
    """Train a neural network on the training set."""

    model.train()
    train_loss = 0
    correct = 0

    outputs_all = []

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()   # zero the parameter gradients
        input_features, labels = batch_to_torch(data['input_features']), data['label'].type(torch.FloatTensor)
        # embed()
        output = model(input_features)
        output = np.squeeze(output, axis=1)

        loss = criterion(output, labels)
        loss.backward()         # passes the loss backwards to compute the dE/dW gradients
        optimizer.step()        # update our weights

        # evaluate performance
        train_loss += loss.item()
        if (torch.abs(output-labels) < args.correct_threshold).all():
            correct += 1

        output_np = output.detach().numpy()
        outputs_all.append(output_np)

    train_loss /= len(train_loader.dataset)
    accuracy = (correct / len(train_loader.dataset)) * 100

    return train_loss, accuracy, outputs_all


def test(args, model, device, test_loader, criterion, printOutput=True):
    """
    Test a neural network on the test set.
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # dont track the gradients
        for batch_idx, data in enumerate(test_loader):
            input_features, labels = batch_to_torch(data['input_features']), data['label'].type(torch.FloatTensor)
            output = model(input_features)
            output = np.squeeze(output, axis=1)
            test_loss += criterion(output, labels).item()

            if (torch.abs(output-labels) < args.correct_threshold).all():
                correct += 1

    test_loss /= len(test_loader.dataset)
    accuracy = (correct / len(test_loader.dataset)) * 100
    if printOutput:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy


class Net(nn.Module):
    """
    This is a network with the 'inverse' architecture to the one in Saxe et al. (2019) PNAS.
    """
    def __init__(self, D_features_in, D_out, D_h_features):
        super(Net, self).__init__()
        self.h_features_size = D_h_features

        self.fc_feature_to_hfeatures = nn.Linear(D_features_in, self.h_features_size)  # size input, size output
        self.fc_hfeature_to_out  = nn.Linear(self.h_features_size, D_out)

    def forward(self, x_feature):

        # just linear activations with MSE loss:
        self.hfeature_activations = self.fc_feature_to_hfeatures(x_feature)
        self.output = self.fc_hfeature_to_out(self.hfeature_activations)

        return self.output

    def get_activations(self, x_feature):
        self.forward(x_feature)  # update the activations with the particular input
        return self.hfeature_activations


def init_weights(m):
    """Initialise network weights with custom settings.
      - Keep gain small to ensure small and low variance weights.
      """
    if type(m) == nn.Linear:
        gain = 1e-5
        torch.nn.init.xavier_uniform(m.weight, gain)


def print_progress(i, numiter):
    """This function prints to the screen the optimisation progress (at each iteration i, out of a total of numiter iterations)."""
    j = i/numiter
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% " % ('-'*int(20*j), 100*j))
    sys.stdout.flush()


def log_performance(writer, epoch, train_loss, test_loss, train_accuracy, test_accuracy):
    """ Write out the training and testing performance for this epoch to tensorboard.
          - 'writer' is a SummaryWriter instance
    """
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)


def get_activations(args, trained_model, test_loader):
    """ This will determine the hidden unit activations for each input pair in the train/test set."""

    #  pass each input through the network and see what happens to the hidden layer activations
    item_activations = np.zeros((args.n_unique, args.D_h_features)) # should strictly be called sth like feature_activations
    trained_model.eval()
    with torch.no_grad():
        for sample_idx, data in enumerate(test_loader):
            input_features, labels = batch_to_torch(data['input_features']), data['label'].type(torch.FloatTensor)
            item_act = trained_model.get_activations(input_features)
            item_activations[sample_idx] = item_act

    activations = [item_activations]
    return activations


def get_model_name(args):
    """Determine the correct name for the model and analysis files."""
    hiddensizes = '_' + str(args.D_h_features)
    model_name = const.MODEL_DIRECTORY + str(args.id) + '_model' + hiddensizes + '.pth'
    analysis_name = const.ANALYSIS_DIRECTORY + str(args.id) + '_model_analysis' + hiddensizes + '.npy'
    return model_name, analysis_name


def train_network(args, device, trainset, testset, plot_svds=False):
    """
    This function performs the train/test loop for training the Rogers/McClelland '08 analogy model
    """

    print("Network training conditions: ")
    print(args)
    print("\n")

    # Define a model for training
    model = Net(args.D_features_in, args.D_out, args.D_h_features).to(device)
    model.apply(init_weights)

    criterion = nn.MSELoss()   # mean squared error loss
    #criterion = nn.BCELoss() # binary cross entropy loss
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Define our dataloaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Train/test loop
    n_epochs = args.epochs
    printOutput = False

    # Log the model on TensorBoard and label it with the date/time and some other naming string
    now = datetime.now()
    date = now.strftime("_%d-%m-%y_%H-%M-%S")
    comment = "_lr-{}_epoch-{}_hfeatures{}".format(args.lr, args.epochs, args.D_h_features)
    writer = SummaryWriter(log_dir=const.TB_LOG_DIRECTORY + 'record_' + date + comment)
    print("Open tensorboard in another shell to monitor network training (hannahsheahan$  tensorboard --logdir=training_records/tensorboard)")

    train_loss_record, test_loss_record, train_accuracy_record, test_accuracy_record = [[] for i in range(4)]

    train_outputs_record = []

    print("Training network...")

    hidden_activity_all = []
    hidden_distance_all = []
    hidden_activity, hidden_distance = svd_plots.get_hidden_activity_and_distance(args, model, testset, 0)
    hidden_activity_all.append(hidden_activity); hidden_distance_all.append(hidden_distance)
    for epoch in range(1, n_epochs + 1):  # loop through the whole dataset this many times

        # train network
        train_loss, train_accuracy, train_outputs = train(args, model, device, trainloader, optimizer, criterion, epoch, printOutput)

        # assess network
        test_loss, test_accuracy = test(args, model, device, testloader, criterion, printOutput)

        # log performance
        log_performance(writer, epoch, train_loss, test_loss, train_accuracy, test_accuracy)
        if epoch % args.log_interval == 0:
            print('loss: {:.10f}, accuracy: {:.1f}%'.format(train_loss, train_accuracy))
            print_progress(epoch, n_epochs)

        train_accuracy_record.append(train_accuracy)
        test_accuracy_record.append(test_accuracy)
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)

        train_outputs = [train_outputs[i].tolist() for i in range(len(train_outputs))]
        train_outputs_record.append(train_outputs)
        # if epoch % 50 == 0:
        hidden_activity, hidden_distance = svd_plots.get_hidden_activity_and_distance(args, model, testset, epoch)
        hidden_activity_all.append(hidden_activity); hidden_distance_all.append(hidden_distance)
        # z_scored_hidden_distance = zscore(hidden_distance)


    if plot_svds:
        svd_plots.plot_sing_val_trajectory(args, train_outputs_record)

    # svd_plots.plot_RDMs(args, hidden_activity_all, hidden_distance_all)
    # svd_plots.plot_RDM_regression(args, hidden_distance_all)

    try:
        hidden_distance_all = [[hidden_distance_all[i][0].tolist(), hidden_distance_all[i][1].tolist(), hidden_distance_all[i][2].tolist(), hidden_distance_all[i][3].tolist()] for i in range(len(hidden_distance_all))]
    except:
        embed()

    record = {"train_loss":train_loss_record, "test_loss":test_loss_record, "train_accuracy":train_accuracy_record, "test_accuracy":test_accuracy_record, "train_outputs":train_outputs_record, 'hidden_distances': hidden_distance_all, "args":vars(args) }
    randnum = str(random.randint(0,10000))
    args.id = randnum
    model_name, _ = get_model_name(args)
    dat = json.dumps(record)
    path = const.TRAININGRECORDS_DIRECTORY + randnum + date + comment + ".json"
    f = open(path,"w")
    f.write(dat)
    f.close()

    writer.close()
    print("Training complete.")

    print('\nSaving trained model...')
    print(model_name)
    torch.save(model, model_name)

    return model, randnum, path
