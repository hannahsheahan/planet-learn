import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcol
from IPython import embed
from torch.utils.data import Dataset, DataLoader
import network as net
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import glob
import os
import json
import seaborn as sns; sns.set(); sns.set_style(style='white')
import pandas as pd


def plots_multiple_runs():
    runs = 5
    for run in range(runs):
        # call network and return all that I need for making those plots
        pass

def get_RDM_regression_coeffs(hidden_distance):

    model_RDM = np.array([[0, 10, 10, 10],
                          [10, 0, 10, 10],
                          [10, 10, 0, 10],
                          [10, 10, 10, 0]])
    model_RDM = zscore(model_RDM)
    # print('\n \n Model RDM z scored \n')
    # print(model_RDM)

    coeff_array = []#; bias_array = []

    for epoch in range((len(hidden_distance))):
        z_scored_hidden_distance = zscore(hidden_distance[epoch])
        X = np.vstack((np.ones((1,16)),np.ravel(z_scored_hidden_distance))).T
        Y = np.ravel(model_RDM)
        reg = LinearRegression().fit(X, Y)

        coeff_array.append(reg.coef_[1])
        # bias_array.append(reg.coef_[0])

    return coeff_array
    # fig, ax = plt.subplots(1,1)
    # ax.plot(coeff_array)
    # ax.set_xlabel('epochs')
    # ax.set_ylabel('coefficient')

    # # ax_bias.plot(bias_array)
    # # ax_bias.set_xlabel('epochs')
    # # ax_bias.set_ylabel('bias coeffs')
    # plt.show()


def get_sing_vals(train_outputs):
    nb_of_nets = len(train_outputs)
    epochs = len(train_outputs[0]) # args.epochs
    for net_idx in range(nb_of_nets):
        assert epochs == len(train_outputs[net_idx]), "nets trained on different epoch numbers, error raised when evaluating net " + str(net_idx) + ', func: plot_sing_val_trajectory'
    # assert epochs == len(train_outputs)
    nb_items = 4 # args.D_out

    inputs = np.array([[1, 1, 0, 1, 0, 0, 0],
                  [1, 1, 0, 0, 1, 0, 0],
                  [1, 0, 1, 0, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 1]])

    # embed()
    A_vals = []
    shell = True

    # plot_after_epochs = [100, 200, 300, 400, 500, 600]
    # plot_after_epochs = [150, 300, 450, 600, 750, 900]
    plot_after_epochs = [int(np.floor((epochs/6)*i)) for i in range(1,7)]
    data_plot_svds = {}
    fig, axs = plt.subplots(1,1)
    for net in range(nb_of_nets):
        data_plot_svds[net] = {}
        A_vals.append([])
        for epoch in range(1, epochs+1):
            try:
            	outputs_transformed = np.array([np.array(train_outputs[net][epoch-1][i]).reshape(4,1) for i in range(nb_items)])
            except:
            	if shell:
            		embed()
            	shell = False
            tmp = [outputs_transformed[i] @ np.array(inputs[i]).reshape(1,7) for i in range(4)]
            approx_cov_matr = np.zeros((4,7))
            for i in range(nb_items):
            	approx_cov_matr += tmp[i]

            U, A, V = svd(approx_cov_matr,full_matrices=False)
            A_vals[net].append(A)

            # if epoch in plot_after_epochs: #np.floor(epochs/2):
            data_plot_svds[net][epoch] = [approx_cov_matr, U, A, V]

    return A_vals, data_plot_svds

def plot_sing_val_trajectory(A_vals, axs, plot_after_epochs, plot_average=False, show_plot=True):
    colors = ['blue', 'green', 'red', 'orange', 'black', 'grey', 'olive']
    if plot_average:
        for net_idx in range(len(A_vals)):
            DF_a_vals_net = pd.DataFrame()
            DF_a_vals_net['net_idx'] = np.ravel(np.ones((1, len(A_vals[net_idx]))) * net_idx)
            DF_a_vals_net['epoch']   = list(range(len(A_vals[net_idx])))
            DF_a_vals_net['a1']      = np.array([A_vals[net_idx][i][0] for i in range(len(A_vals[net_idx]))])
            DF_a_vals_net['a2']      = np.array([A_vals[net_idx][i][1] for i in range(len(A_vals[net_idx]))])
            DF_a_vals_net['a3']      = np.array([A_vals[net_idx][i][2] for i in range(len(A_vals[net_idx]))])
            DF_a_vals_net['a4']      = np.array([A_vals[net_idx][i][3] for i in range(len(A_vals[net_idx]))])
        
            if net_idx == 0:
                DF_a_vals = DF_a_vals_net
            else:
                DF_a_vals = DF_a_vals.append(DF_a_vals_net,ignore_index=True, sort=True)

        sns.lineplot(data=DF_a_vals, x='epoch', y='a1', ax=axs, ci=68)
        print('plot 1 done')
        sns.lineplot(data=DF_a_vals, x='epoch', y='a2', ax=axs, ci=68)
        print('plot 2 done')
        sns.lineplot(data=DF_a_vals, x='epoch', y='a3', ax=axs, ci=68)
        print('plot 3 done')
        sns.lineplot(data=DF_a_vals, x='epoch', y='a4', ax=axs, ci=68)
        print('plot 4 done')

        axs.set_ylabel('a_i(t)')
        try:
            ymax = np.amax(DF_a_vals_net['a1'].values)
        except Exception as e:
            ymax = 2.7
            print('error.\n')
            print(str(e))
        axs.vlines(x=plot_after_epochs[1:], ymin=0, ymax=ymax, colors='black', linestyles='dashed')


    else:
        # choose color scheme:
        color_sing_values = False
        if color_sing_values:
            colors_plot = [[colors[0], colors[1], colors[2], colors[3]] for i in range(len(A_vals))] # consistent colors for equivalent singular values:
            labels_plot = ['a1', 'a2', 'a3', 'a4'] + [[""]*4 for i in range(len(A_vals)-1)]
        else:
            colors_plot = [[colors[net_idx]]*4 for net_idx in range(len(A_vals))]# consistent color for same net
            labels_plot = [['net' + str(i), "", "", ""] for i in range(len(A_vals))]
        for net in range(len(A_vals)):

            # PLOT TRAJECTORY OF SINGULAR VALUES
            a1 = [A_vals[net][i][0] for i in range(len(A_vals[net]))]
            a2 = [A_vals[net][i][1] for i in range(len(A_vals[net]))]
            a3 = [A_vals[net][i][2] for i in range(len(A_vals[net]))]
            a4 = [A_vals[net][i][3] for i in range(len(A_vals[net]))]

            # # plot in same subplot: 
            
            if net == 0:
                axs.plot(a1, label=labels_plot[net][0], color=colors_plot[net][0])
                axs.plot(a2, label=labels_plot[net][1], color=colors_plot[net][1])
                axs.plot(a3, label=labels_plot[net][2], color=colors_plot[net][2])
                axs.plot(a4, label=labels_plot[net][3], color=colors_plot[net][3])
            else:
                axs.plot(a1, label=labels_plot[net][0], color=colors_plot[net][0])
                axs.plot(a2, label=labels_plot[net][1], color=colors_plot[net][1])
                axs.plot(a3, label=labels_plot[net][2], color=colors_plot[net][2])
                axs.plot(a4, label=labels_plot[net][3], color=colors_plot[net][3])

            # plot in different subplots:
            # axs[net].plot(a1, label='a1')
            # axs[net].plot(a2, label='a2')
            # axs[net].plot(a3, label='a3')
            # axs[net].plot(a4, label='a4')

            try:
                ymax = np.amax(a1 + a2 + a3 + a4)
            except Exception as e:
                ymax = 2.7
                print('error.\n')
                print(str(e))
            # axs[net].vlines(x=plot_after_epochs, ymin=0, ymax=ymax, colors='black', linestyles='dashed')
            # axs[net].set_ylabel('a_i(t)')
            # axs[net].set_xlabel('t(Epochs)')

            # axs.vlines(x=plot_after_epochs, ymin=0, ymax=ymax, colors='black', linestyles='dashed')
            axs.set_ylabel('a_i(t)')
            axs.set_xlabel('t(Epochs)')
            axs.legend(loc='lower right')

        if show_plot:
            plt.show()

        # print('\n shape A')
        # print(np.array(A_vals).shape)

        # REPRODUCE ANDREW'S FIGURE 3B:
        # reproduce_Fig3B(data_plot_svds, plot_after_epochs)

def reproduce_Fig3B(data_plot_svds, plot_after_epochs):
    # data plot svds: net x epoch x [approx_cov_matr, U, A, V] --> convert to: epoch x [approx_cov_matr x net, U x net, A x net, V x net]
    reshaped_data_plot_svds = {}
    if plot_after_epochs[0] == 0:
        plot_after_epochs = plot_after_epochs[1:]
    shell = True
    for count, epoch in enumerate(plot_after_epochs):
        reshaped_data_plot_svds[count] = np.array([0,0,0,0])
        this_epoch = []
        for matr in range(len(data_plot_svds[0][epoch])):
            tmp = np.stack([data_plot_svds[net_idx][epoch][matr] for net_idx in data_plot_svds.keys()])
            this_epoch.append(tmp.mean(axis=0))
        reshaped_data_plot_svds[count] = this_epoch

    fig, axs = plt.subplots(len(reshaped_data_plot_svds.keys()),4, figsize=(15,10))
    _min, _max = -1, 1
    for count, plot_nb in enumerate(reshaped_data_plot_svds.keys()):
        [approx_cov_matr, U, A, V] = reshaped_data_plot_svds[plot_nb]
        axs[count][0].imshow(approx_cov_matr, cmap=cm.PuOr, vmin = _min, vmax = _max)
        axs[count][0].set_title('Sigma-hat(t)')
        axs[count][1].imshow(U, cmap=cm.PuOr, vmin = _min, vmax = _max)
        axs[count][1].set_title('U')
        axs[count][2].imshow(np.diag(A), cmap=cm.PuOr, vmin = _min, vmax = _max)
        axs[count][2].set_title('A(t)')
        im = axs[count][3].imshow(V, cmap=cm.PuOr, vmin = _min, vmax = _max)
        axs[count][3].set_title('V')    
            # plt.title('After ' + str(plot_nb) ' training epochs')

            # matrx = U@np.diag(A)@V

            # ax5.imshow(matrx, cmap=cm.PuOr, vmin = _min, vmax = _max)
    fig.colorbar(im)
    title = 'After {}, {}, {}, {}, {} epochs (from top to bottom)'.format(*plot_after_epochs)
    fig.suptitle(title)
    plt.subplots_adjust(hspace = 0.6)
    plt.show()
    # plt.subplots_adjust(hspace=0.6)
    # plt.subplots_adjust(hspace=0.6)


def get_hidden_activity_and_distance(args, model, testset, epoch_nb):
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)
    all_activations = net.get_activations(args, model, test_loader)

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

        # plt.figure(figsize=(10,10.5))
        hidden_activity = np.zeros((args.n_items, hdim))
        for input_idx in range(args.n_inputs):
            count = input_idx # int(lookup[input_idx])
            attr_idx = input_idx
            try:
                hidden_activity[attr_idx, :] = activations[count, :].flatten()
            except:
                embed()

            # compute distance matrix over hidden activations
            # if a and input_idx == 3:
            #     embed()
            hidden_distance = pairwise_distances(hidden_activity, metric='euclidean')

            np.fill_diagonal(np.asarray(hidden_distance), 0)
            total_hidden_activity[:,:] = hidden_activity
            total_hidden_distance[:,:] = hidden_distance

    return hidden_activity, hidden_distance


def plot_hidden_activity_RDMs(hidden_activity):
    nb_of_nets = len(hidden_activity)
    epochs = len(hidden_activity[0]) # args.epochs
    for net_idx in range(nb_of_nets):
        assert epochs == len(hidden_activity[net_idx]), "nets trained on different epoch numbers, error raised when evaluating net " + str(net_idx) + ", func: plot_hidden_activity_RDMs."

    epochs_betw_plots = 500
    n_plots = np.ceil(epochs/epochs_betw_plots) + 1
    n_cols  = 4 
    n_rows  = int(np.ceil(n_plots/n_cols))

    fig, axs  = plt.subplots(n_rows,n_cols,figsize=(15,10))
    _min, _max = -1, 1

    for count, matr in enumerate(hidden_activity):
        row = int(np.floor(count/n_cols))
        col = int(count%n_cols)
        im = axs[row][col].imshow(matr, cmap=cm.PuOr, vmin = _min, vmax = _max)
        axs[row][col].set_title('features hidden layer after ' + str(count*epochs_betw_plots) + ' epochs')
        axs[row][col].set_xlabel('hidden units')
        # plt.ylabel('items x domains')
        axs[row][col].set_ylabel('items only')
    fig.colorbar(im)
    plt.show()  
    



def plot_RDMs_hidden_distance(hidden_distance, plot_after_epochs):
    nb_of_nets = len(hidden_distance)
    epochs = len(hidden_distance[0])
    for net_idx in range(nb_of_nets):
        assert epochs == len(hidden_distance[net_idx]), "nets trained on different epoch numbers, error raised when evaluating net " + str(net_idx) + ", func: plot_RDMs_hidden_distance."
    # convert list to numpy:
    for net_idx in range(nb_of_nets):
        hidden_distance[net_idx] = np.array(hidden_distance[net_idx])
        for epoch in range(epochs):
            hidden_distance[net_idx][epoch] = np.array(hidden_distance[net_idx][epoch]).reshape(4,4)
    # take average if multiple nets:
    hidden_distance_all_nets = np.array(hidden_distance).mean(axis=0)

    # epochs_betw_plots = 500
    # plot_after_epochs = [1] + [i for i in range(1, epochs+1) if i % epochs_betw_plots == 0]
    # plot_after_epochs[-1] = epochs - 1
    n_plots = len(plot_after_epochs)# np.ceil(epochs/epochs_betw_plots) + 1
    n_cols  = 3
    n_rows  = int(np.ceil(n_plots/n_cols))

    # fig, axs  = plt.subplots(n_rows,n_cols,figsize=(15,10))
    # _min, _max = -1, 1

    
    # NEW PLOT FOR HIDDEN DISTANCE:
    # fig, ax = plt.subplots(1,1)
    # im = ax.imshow(hidden_distance[0], cmap=cm.PuOr)
    # fig.colorbar(im)
    # plt.show()

    # embed()
    fig, axs  = plt.subplots(n_rows,n_cols,figsize=(15,10))
    for count, epoch in enumerate(plot_after_epochs):
        matr = hidden_distance_all_nets[epoch]
        row = int(np.floor(count/n_cols))
        col = int(count%n_cols)
        im = axs[row][col].imshow(matr, cmap=cm.PuOr)
        # axs[row][col].set_title('features hidden RDM after ' + str(count*50) + ' epochs')
        axs[row][col].set_title('after ' + str(plot_after_epochs[count]) + ' epochs')
        axs[row][col].set_xlabel('items')
        axs[row][col].set_ylabel('items')
        fig.colorbar(im, ax=axs[row][col])
    title = 'Hidden distance RDMs averaged over ' + str(nb_of_nets) + ' nets' if nb_of_nets > 1 else 'Hidden distance RDMs single network'
    plt.suptitle(title)
    plt.subplots_adjust(hspace=0.7)
    plt.show()



def plot_regression_coeffs(DF_all_nets, plot_after_epochs):
    colors = ['blue', 'green', 'red', 'orange', 'black', 'grey', 'olive']

    nb_of_nets = int(DF_all_nets['net_idx'].values[-1])+1

    fig, axs = plt.subplots(1,2)
    sns.lineplot(data=DF_all_nets,x='epoch', y='coeffs', err_style='band', ci=68, ax=axs[0])

    # fig, ax = plt.subplots(1,1)
    plot_individual_lines = False
    plot_sing_vals        = True
    if plot_individual_lines:
        for net in range(nb_of_nets):
            coeffs = DF_all_nets.coeffs.values[(3000*net):(3000*(net+1))]
            axs[0].plot(coeffs, label='net'+str(net), color=colors[net])
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('coefficient')
        axs[0].legend(loc='lower right')
    
    ymax = 1
    ymin = 0.62
    axs[0].vlines(x=plot_after_epochs[1:], ymin=ymin, ymax=ymax, colors='black', linestyles='dashed')

    if plot_sing_vals:
        train_outputs = [DF_all_nets[DF_all_nets['net_idx']==net].train_outputs.values for net in range(nb_of_nets)]
        sing_vals, data_plot_svds = get_sing_vals(train_outputs)
        plot_sing_val_trajectory(sing_vals, axs[1], plot_after_epochs=plot_after_epochs, plot_average = True, show_plot=False)

    # fig.suptitle('')
    axs[0].set_title('regression coefficients model RDM - hidden rep RDM')
    axs[1].set_title('singular values evolving over time (averaged over 5 nets)')
    plt.show()


def getTrainRecordsFromJSON():

    nb_of_nets = 5
    list_of_files = glob.glob('training_records/*.json')
    analysis_data = []
    for i in range(nb_of_nets):
        newest_file = max(list_of_files, key=os.path.getctime)
        analysis_data.append(newest_file)
        del(list_of_files[list_of_files.index(newest_file)])

    hidden_distance_all_nets, train_outputs_all_nets = [], []
    for file in analysis_data:
        open_file = open(file)
        text = open_file.read()
        data = json.loads(text)
        hidden_distance_all_nets.append(data['hidden_distances'])
        train_outputs_all_nets.append(data['train_outputs'])

    for net_idx in range(len(hidden_distance_all_nets)):
        coeffs = get_RDM_regression_coeffs(hidden_distance_all_nets[net_idx])
        DF_net = pd.DataFrame()
        DF_net['net_idx'] = np.ravel(np.ones((1, len(coeffs[1:]))) * net_idx)
        DF_net['epoch']  = list(range(len(coeffs[1:])))
        DF_net['coeffs'] = coeffs[1:]
        DF_net['train_outputs'] = train_outputs_all_nets[net_idx]

        if net_idx == 0:
            DF_all_nets = DF_net

        else:
            DF_all_nets = DF_all_nets.append(DF_net,ignore_index=True, sort=True)

    print('analysed data from %d nets'%(net_idx+1))

    # epochs_betw_plots = 500
    # plot_after_epochs = [1] + [i for i in range(1, epochs+1) if i % epochs_betw_plots == 0]
    # plot_after_epochs[-1] = epochs - 1
    plot_after_epochs = [0, 200, 1100, 1800, 2400, 2800]

    # PLOT REGRESSION COEFFS:
    # plot_regression_coeffs(DF_all_nets, plot_after_epochs)

    train_outputs = [DF_all_nets[DF_all_nets['net_idx']==net].train_outputs.values for net in range(nb_of_nets)]
    A_vals, data_plot_svds = get_sing_vals(train_outputs)
    # plot_sing_val_trajectory(train_outputs)

    # plot_RDMs_hidden_distance(hidden_distance_all_nets, plot_after_epochs)

    # embed()
    # print(analysis_data)
    # embed()
    reproduce_Fig3B(data_plot_svds, plot_after_epochs)

# getTrainRecordsFromJSON()

if __name__ == '__main__':
    getTrainRecordsFromJSON()




