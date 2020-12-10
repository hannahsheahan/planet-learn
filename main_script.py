"""
'Inverse' model of Saxe et al 2019 PNAS
Author: Hannah Sheahan, sheahan.hannah@gmail.com & Fabian Otto, fabianotto267@gmail.com
Date: 28/11/2020
Notes: N/A
Issues: N/A
"""
# ---------------------------------------------------------------------------- #
import numpy as np
import sys
from config import get_config
import network as net
import matplotlib.pyplot as plt
import scipy.io as spio
import constants as const
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from IPython import embed


def setup_inputs(args):
    """ Setup inputs and attributes."""

    # on each trial we have a 3-hot feature input
    # create manually (for now):
    inputs = np.array([[1, 1, 0, 1, 0, 0, 0],
                      [1, 1, 0, 0, 1, 0, 0],
                      [1, 0, 1, 0, 0, 1, 0],
                      [1, 0, 1, 0, 0, 0, 1]])

    lookup = np.array([1,2,3,4])
    
    # Check that unique inputs appropriately made
    plt.figure()
    plt.imshow(inputs)
    plt.title('Features')
    plt.ylabel('input id #')
    plt.savefig(const.FIGURE_DIRECTORY + 'Inputs_coding.pdf',bbox_inches='tight')

    return lookup, inputs


def setup_outputs(args, lookup):
    output_items = np.array([[1,0,0,0],
    				        [0,1,0,0],
    				        [0,0,1,0],
    				        [0,0,0,1]])

    return output_items


def analyse_network(args, trainset, testset, lookup):
    """Analyse the hidden unit activations for each unique input in each context.
    """
    model_name, analysis_name = net.get_model_name(args)

    # load an existing dataset
    try:
        data = np.load(analysis_name, allow_pickle=True)
        MDS_dict = data.item()
        preanalysed = True
        print('\nLoading existing network analysis...')
    except:
        preanalysed = False
        print('\nAnalysing trained network...')

    if not preanalysed:
        # load the trained model and the datasets it was trained/tested on
        trained_model = torch.load(model_name)

        # Assess the network activations on test set
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)
        all_activations = net.get_activations(args, trained_model, test_loader)

        item_activations = all_activations
        layers = ['features']
        hidden_sizes = [args.D_h_features]

        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            activations = all_activations[layer_idx]
            hdim = hidden_sizes[layer_idx]

            # format into RDM
            total_hidden_distance = np.zeros((args.n_items, args.n_items))
            total_hidden_activity = np.zeros((args.n_items, hdim))

            plt.figure(figsize=(10,10.5))
            hidden_activity = np.zeros((args.n_items, hdim))
            for input_idx in range(args.n_inputs):
                count = input_idx # int(lookup[input_idx])
                attr_idx = input_idx
                try:
                    hidden_activity[attr_idx, :] = activations[count, :].flatten()
                except:
                    embed()

                # compute distance matrix over hidden activations
                hidden_distance = pairwise_distances(hidden_activity, metric='euclidean')
                np.fill_diagonal(np.asarray(hidden_distance), 0)
                total_hidden_activity[:,:] = hidden_activity
                total_hidden_distance[:,:] = hidden_distance

                # embed()

                # plt.subplot(2,2,context_idx * args.n_contexts + 1)
                # print('hidden activity \n')
                # print(hidden_activity)
                plt.imshow(hidden_activity)
                plt.colorbar
                plt.title(layer + 'hidden layer')
                plt.xlabel('hidden units')
                # plt.ylabel('items x domains')
                plt.ylabel('items only')

                # plt.subplot(2,2,context_idx * args.n_contexts + 2)
                # print('hidden distance \n')
                # print(hidden_distance)
                plt.imshow(hidden_distance)
                plt.colorbar
                plt.title(layer + 'hidden RDM')
                # plt.xlabel('items x domains')
                # plt.ylabel('items x domains')
                plt.xlabel('items only')
                plt.ylabel('items only')

            plt.savefig(const.FIGURE_DIRECTORY + layer + 'hidden_activity_RDMs'+model_name[7:-4]+'.pdf',bbox_inches='tight')



def plot_learning_curve(args, json_path):
    """Get the record of training and plot loss and accuracy over time."""
    #model_name, analysis_name = net.get_model_name(args)

    #record_name = '8083_23-11-20_17-13-17lr-0.05_epochs-30000.json'
    # record_name = json_path
    # with open(os.path.join(const.TRAININGRECORDS_DIRECTORY, record_name)) as record:
    #     data = json.load(record)

    record_name = json_path[len(const.TRAININGRECORDS_DIRECTORY):]

    with open(json_path) as record:
        data = json.load(record)


    plt.figure()
    plt.plot(data['train_loss'])
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.savefig(const.FIGURE_DIRECTORY + 'traintraj_' + record_name[:-5] + '.pdf', bbox_inches='tight')


def main():
    args, device = get_config()

    # load in the inputs and outputs I built in matlab (before realising I really want to train this in pytorch)
    lookup, inputs = setup_inputs(args)
    outputs = setup_outputs(args, lookup)

    # define train and test sets using our Dataset-inherited class
    dataset = {'index':list(range(args.n_unique)),'input_features':inputs,'label':outputs}
    trainset = net.CreateDataset(dataset)
    testset = net.CreateDataset(dataset)  # HRS note that, for now, train and test are same dataset. As in Rogers/McClelland

    # embed()
    # train and test network
    model, id, json_path = net.train_network(args, device, trainset, testset)#, plot_svds=True)
    args.id = id
    # analyse trained network hidden activations
    analyse_network(args, trainset, testset, lookup)

    # plot training record and save it
    plot_learning_curve(args, json_path)


main()
