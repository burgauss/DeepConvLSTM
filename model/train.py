import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from misc.torchutils import count_parameters, seed_worker
from config import config
from torch import nn
from model.models import DeepConvLSTM, ConvBlock, ConvBlockSkip, ConvBlockFixup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
import os


def init_weights(network):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network.

    :param network: pytorch model
        Network of which weights are to be initialised
    :return: pytorch model
        Network with initialised weights
    """
    for m in network.modules():
        # normal convblock and skip convblock initialisation
        if isinstance(m, (ConvBlock, ConvBlockSkip)):
            if network.weights_init == 'normal':
                torch.nn.init.normal_(m.conv1.weight)
                torch.nn.init.normal_(m.conv2.weight)
            elif network.weights_init == 'orthogonal':
                torch.nn.init.orthogonal_(m.conv1.weight)
                torch.nn.init.orthogonal_(m.conv2.weight)
            elif network.weights_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.conv1.weight)
                torch.nn.init.xavier_uniform_(m.conv2.weight)
            elif network.weights_init == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.conv1.weight)
                torch.nn.init.xavier_normal_(m.conv2.weight)
            elif network.weights_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.conv1.weight)
                torch.nn.init.kaiming_uniform_(m.conv2.weight)
            elif network.weights_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.conv1.weight)
                torch.nn.init.kaiming_normal_(m.conv2.weight)
            m.conv1.bias.data.fill_(0.0)
            m.conv2.bias.data.fill_(0.0)
        # fixup block initialisation (see fixup paper for details)
        elif isinstance(m, ConvBlockFixup):
            nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * network.nb_conv_blocks ** (-0.5))
            nn.init.constant_(m.conv2.weight, 0)
        # linear layers
        elif isinstance(m, nn.Linear):
            if network.use_fixup:
                nn.init.constant_(m.weight, 0)
            elif network.weights_init == 'normal':
                torch.nn.init.normal_(m.weight)
            elif network.weights_init == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight)
            elif network.weights_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight)
            elif network.weights_init == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight)
            elif network.weights_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight)
            elif network.weights_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # LSTM initialisation
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    if network.weights_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif network.weights_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif network.weights_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif network.weights_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif network.weights_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif network.weights_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    if network.weights_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif network.weights_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif network.weights_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif network.weights_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif network.weights_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif network.weights_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
    return network



def train_simplified(train_features, train_labels, val_features, val_labels,
        network, optimizer, loss, config, log_date, log_timestamp):
    """ function trains a DeepConvLSTM_Simplified from models.py
    params:
        train_features: np.array
            a normalized X_train dataset configured to work in the gpu. example: X_train_ss.astype(np.float32), y_train.astype(np.uint8)
        train_labels: np.array
            a normalized or unnormalized X_train label dataset configured to work in the gpu. See example above
        val_features : np.array
            a normalized validation dataset. NOT USED
        val_labels: np.array
            a normalized or unnormalized y_val label dataset configured to work in the gpu. See example above. 
            NOT USED
        network: the model defined in pytorch, in this case the DeepConvLSTM_simplified
        optimizer: an optimizer from pytorch
        loss: a loss calculation from pytorch
        config: dictionary
            a dictionary having the characteristics of the dataset and the model
        log_date: time object
            not used
        log_timestamp: time object
            not used
    returns:
        network: the trained model"""

    config['window_size'] = train_features.shape[1]
    config['nb_channels'] = train_features.shape[2]

    network.to(config['gpu'])
    network.train()

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    
    trainLoader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True)
    valLoader = DataLoader(valid_dataset, batch_size = config['batch_size'], shuffle=True)
    
    # print("Size ", len(trainLoader))
    # for x,y in trainLoader:
    #     print("Shape X", x.shape)
    #     print("Shape y", y.shape)
    #     break

    optimizer, criterion = optimizer, loss

    for e in range(config['epochs']):
        train_losses = []
        train_preds = []
        train_gt = []
        start_time = time.time()
        batch_num = 1

        for i, (x,y) in enumerate(trainLoader):
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
            optimizer.zero_grad()

            #forward
            train_output = network(inputs)

            #Calculate loss
            # loss = criterion(train_output, targets.long())
            loss = criterion(train_output, targets)

            #Backprop
            loss.backward()
            optimizer.step()

            train_output = torch.nn.functional.softmax(train_output, dim=1)

            train_losses.append(loss.item())

            #create predictions and true labels
            y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
            train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))

            if batch_num % 10 == 0 and batch_num > 0:
                cur_loss = np.mean(train_losses)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | train loss {:5.2f}'.format(e, batch_num, elapsed * 1000 / config['batch_size'], cur_loss))
                start_time = time.time()
            batch_num += 1

    return network

def train_validate_simplified(train_features, train_labels, val_features, val_labels,
                    network, optimizer, loss, log_date, log_timestamp):
    
    """Function that trains and at the same time validates the network as defined in
    DeepConvLSTM_simplified
    params:
        train_features: np.array
            a normalized X_train dataset configured to work in the gpu. example: X_train_ss.astype(np.float32), y_train.astype(np.uint8)
        train_labels: np.array
            a normalized or unnormalized X_train label dataset configured to work in the gpu. See example above
        val_features : np.array
            a normalized validation dataset. see example above
        val_labels: np.array
            a normalized or unnormalized y_val label dataset configured to work in the gpu. See example above
        network: the model defined in pytorch, in this case the DeepConvLSTM_simplified
        optimizer: an optimizer from pytorch
        loss: a loss calculation from pytorch
        config: dictionary
            a dictionary having the characteristics of the dataset and the model
        log_date: time object
            not used
        log_timestamp: time object
            not used
    returns:
        network: the trained model
    """
    config['window_size'] = train_features.shape[1]
    config['nb_channels'] = train_features.shape[2]

    network.to(config['gpu'])
    network.train()

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    
    trainLoader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True)
    valLoader = DataLoader(valid_dataset, batch_size = config['batch_size'], shuffle=True)
    
    # print("Size ", len(trainLoader))
    # for x,y in trainLoader:
    #     print("Shape X", x.shape)
    #     print("Shape y", y.shape)
    #     break

    optimizer, criterion = optimizer, loss

    for e in range(config['epochs']):
        train_losses = []
        train_preds = []
        train_gt = []
        start_time = time.time()
        batch_num = 1

        for i, (x,y) in enumerate(trainLoader):
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
            optimizer.zero_grad()

            #forward
            train_output = network(inputs)

            #Calculate loss
            # loss = criterion(train_output, targets.long())
            loss = criterion(train_output, targets)

            #Backprop
            loss.backward()
            optimizer.step()

            train_output = torch.nn.functional.softmax(train_output, dim=1)

            train_losses.append(loss.item())

            #create predictions and true labels
            y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
            train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))

            if batch_num % 10 == 0 and batch_num > 0:
                cur_loss = np.mean(train_losses)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | train loss {:5.2f}'.format(e, batch_num, elapsed * 1000 / config['batch_size'], cur_loss))
                start_time = time.time()
            batch_num += 1

        val_preds = []
        val_gt = []
        val_losses = []

        network.eval()
        with torch.no_grad():
            # iterate over the valloader object
            for i, (x, y) in enumerate(valLoader):
                #again prepare to use GPU
                inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

                #send inputs through network to get predictions
                val_output = network(inputs)
                
                #calculates loss by passing criterion both predictions and true labels
                val_loss = criterion(val_output, targets)

                #Calculate actual prediction (i.e. softmax probabilities)
                val_output = torch.nn.functional.softmax(val_output, dim=1)

                #appends validation loss to list
                val_losses.append(val_loss.item())

                # create predictions and true labels; appends them to the final lists
                y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
                y_true = targets.cpu().numpy().flatten()
                val_preds = np.concatenate((np.array(val_preds, int), np.array(y_preds, int)))
                val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))

            print("\nEPOCH: {}/{}".format(e + 1, config['epochs']),
                "\nTrain Loss: {:.4f}".format(np.mean(train_losses)),
                "Train Acc: {:.4f}".format(jaccard_score(train_gt, train_preds, average='macro')),
                "Train Prec: {:.4f}".format(precision_score(train_gt, train_preds, average='macro')),
                "Train Rcll: {:.4f}".format(recall_score(train_gt, train_preds, average='macro')),
                "Train F1: {:.4f}".format(f1_score(train_gt, train_preds, average='macro')),
                "\nVal Loss: {:.4f}".format(np.mean(val_losses)),
                "Val Acc: {:.4f}".format(jaccard_score(val_gt, val_preds, average='macro')),
                "Val Prec: {:.4f}".format(precision_score(val_gt, val_preds, average='macro')),
                "Val Rcll: {:.4f}".format(recall_score(val_gt, val_preds, average='macro')),
                "Val F1: {:.4f}".format(f1_score(val_gt, val_preds, average='macro')))
        
        network.train()

    return network

def plot_grad_flow(network):
    """
    Function which plots the average gradient of a network.

    :param network: pytorch model
        Network used to obtain gradient
    """
    named_parameters = network.named_parameters()
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show


def train(train_features, train_labels, val_features, val_labels, network, optimizer,
         loss, lr_scheduler=None, log_dir=None):
    """
    Method to train a PyTorch network.

    :param train_features: numpy array
        Training features
    :param train_labels: numpy array
        Training labels
    :param val_features: numpy array
        Validation features
    :param val_labels: numpy array
        Validation labels
    :param network: pytorch model
        DeepConvLSTM network object
    :param optimizer: optimizer object
        Optimizer object
    :param loss: loss object
        Loss object
    :param config: dict
        Config file which contains all training and hyperparameter settings
    :param log_date: string
        Date used for logging
    :param log_timestamp: string
        Timestamp used for logging
    :param lr_scheduler: scheduler object, default: None
        Learning rate scheduler object
    :return pytorch model, numpy array, numpy array
        Trained network and training and validation predictions with ground truth
    """

    # prints the number of learnable parameters in the network
    count_parameters(network)

    # init network using weight initialization of choice
    network = init_weights(network)
    # send network to GPU
    network.to(config['gpu'])
    network.train()

    # if weighted loss chosen, calculate weights based on training dataset; else each class is weighted equally
    if config['weighted']:
        # class_weights = torch.from_numpy(
        #     # 09122022 In train no class 0 exists, so we comment out the next line
        #     #compute_class_weight('balanced', classes=np.unique(train_labels + 1), y=train_labels + 1)).float()
        #     compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)).float()
        # if config['loss'] == 'cross_entropy':
        #     loss.weight = class_weights.cuda()
        # print('Applied weighted class weights: ')
        # print(class_weights)
        pass
    
    else:
        class_weights = torch.from_numpy(
            # 09122022 In train no class 0 exists, so we comment out the next line
            compute_class_weight('balanced', classes=np.unique(train_labels + 1), y=train_labels + 1)).float()
            #compute_class_weight(None, classes=np.unique(train_labels), y=train_labels)).float()
        if config['loss'] == 'cross_entropy':
            loss.weight = class_weights.cuda()

    # initialize optimizer and loss
    opt, criterion = optimizer, loss

    # if config['loss'] == 'maxup':
    #     maxup = Maxup(my_noise_addition_augmenter, ntrials=4)

    # initialize training and validation dataset, define DataLoaders
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))

    g = torch.Generator()
    g.manual_seed(config['seed'])

    trainloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffling'],
                             worker_init_fn=seed_worker, generator=g, pin_memory=True)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    valloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                           worker_init_fn=seed_worker, generator=g, pin_memory=True)

    # counters and objects used for early stopping and learning rate adjustment
    best_metric = 0.0
    best_network = None
    best_val_losses = None
    best_train_losses = None
    best_val_preds = None
    best_train_preds = None
    early_stop = False
    es_pt_counter = 0
    labels = list(range(0, config['nb_classes']))

    # training loop; iterates through epochs
    for e in range(config['epochs']):
        """
        TRAINING
        """
        # helper objects
        train_preds = []
        train_gt = []
        train_losses = []
        start_time = time.time()
        batch_num = 1

        # iterate over train dataset
        for i, (x, y) in enumerate(trainloader):
            # send x and y to GPU
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
            # zero accumulated gradients
            opt.zero_grad()

            # if config['loss'] == 'maxup':
            #     # Increase the inputs via data augmentation
            #     inputs, targets = maxup(inputs, targets)

            # send inputs through network to get predictions, calculate loss and backpropagate
            train_output = network(inputs)

            # if config['loss'] == 'maxup':
            #     # calculates loss
            #     train_loss = maxup.maxup_loss(train_output, targets.long())[0]
            # else:
            train_loss = criterion(train_output, targets.long())

            train_loss.backward()
            opt.step()
            # append train loss to list
            train_losses.append(train_loss.item())

            # create predictions and append them to final list
            y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
            train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))

            # if verbose print out batch wise results (batch number, loss and time)
            if config['verbose']:
                if batch_num % config['print_freq'] == 0 and batch_num > 0:
                    cur_loss = np.mean(train_losses)
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | '
                          'train loss {:5.2f}'.format(e, batch_num, elapsed * 1000 / config['batch_size'], cur_loss))
                    start_time = time.time()
                batch_num += 1

            # plot gradient flow if wanted
            if config['save_gradient_plot']:
                plot_grad_flow(network)

        """
        VALIDATION
        """

        # helper objects
        val_preds = []
        val_gt = []
        val_losses = []

        # set network to eval mode
        network.eval()
        with torch.no_grad():
            # iterate over validation dataset
            for i, (x, y) in enumerate(valloader):
                # send x and y to GPU
                inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

                # if config['loss'] == 'maxup':
                #     # Increase the inputs via data augmentation
                #     inputs, targets = maxup(inputs, targets)

                # send inputs through network to get predictions, loss and calculate softmax probabilities
                val_output = network(inputs)
                # if config['loss'] == 'maxup':
                #     # calculates loss
                #     val_loss = maxup.maxup_loss(val_output, targets.long())[0]
                # else:
                val_loss = criterion(val_output, targets.long())

                val_output = torch.nn.functional.softmax(val_output, dim=1)

                # append validation loss to list
                val_losses.append(val_loss.item())

                # create predictions and append them to final list
                y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
                y_true = targets.cpu().numpy().flatten()
                val_preds = np.concatenate((np.array(val_preds, int), np.array(y_preds, int)))
                val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))

            # print epoch evaluation results for train and validation dataset
            print("EPOCH: {}/{}".format(e + 1, config['epochs']),
                  "\nTrain Loss: {:.4f}".format(np.mean(train_losses)),
                  "Train Acc (M): {:.4f}".format(jaccard_score(train_gt, train_preds, average='macro', labels=labels)),
                  "Train Prc (M): {:.4f}".format(precision_score(train_gt, train_preds, average='macro', labels=labels)),
                  "Train Rcl (M): {:.4f}".format(recall_score(train_gt, train_preds, average='macro', labels=labels)),
                  "Train F1 (M): {:.4f}".format(f1_score(train_gt, train_preds, average='macro', labels=labels)),
                  "Train Acc (W): {:.4f}".format(jaccard_score(train_gt, train_preds, average='weighted', labels=labels)),
                  "Train Prc (W): {:.4f}".format(precision_score(train_gt, train_preds, average='weighted', labels=labels)),
                  "Train Rcl (W): {:.4f}".format(recall_score(train_gt, train_preds, average='weighted', labels=labels)),
                  "Train F1 (W): {:.4f}".format(f1_score(train_gt, train_preds, average='weighted', labels=labels)),
                  "\nValid Loss: {:.4f}".format(np.mean(val_losses)),
                  "Valid Acc (M): {:.4f}".format(jaccard_score(val_gt, val_preds, average='macro', labels=labels)),
                  "Valid Prc (M): {:.4f}".format(precision_score(val_gt, val_preds, average='macro', labels=labels)),
                  "Valid Rcl (M): {:.4f}".format(recall_score(val_gt, val_preds, average='macro', labels=labels)),
                  "Valid F1 (M): {:.4f}".format(f1_score(val_gt, val_preds, average='macro', labels=labels)),
                  "Valid Acc (W): {:.4f}".format(jaccard_score(val_gt, val_preds, average='weighted', labels=labels)),
                  "Valid Prc (W): {:.4f}".format(precision_score(val_gt, val_preds, average='weighted', labels=labels)),
                  "Valid Rcl (W): {:.4f}".format(recall_score(val_gt, val_preds, average='weighted', labels=labels)),
                  "Valid F1 (W): {:.4f}".format(f1_score(val_gt, val_preds, average='weighted', labels=labels))
                  )

            # if chosen, print the value counts of the predicted labels for train and validation dataset
            if config['print_counts']:
                y_train = np.bincount(train_preds)
                ii_train = np.nonzero(y_train)[0]
                y_val = np.bincount(val_preds)
                ii_val = np.nonzero(y_val)[0]
                print('Predicted Train Labels: ')
                print(np.vstack((ii_train, y_train[ii_train])).T)
                print('Predicted Val Labels: ')
                print(np.vstack((ii_val, y_val[ii_val])).T)

        # adjust learning rate if enabled
        if config['adj_lr']:
            if config['lr_scheduler'] == 'reduce_lr_on_plateau':
                lr_scheduler.step(np.mean(val_losses))
            else:
                lr_scheduler.step()

        # employ early stopping if employed
        metric = f1_score(val_gt, val_preds, average='macro')
        if best_metric >= metric:
            if config['early_stopping']:
                es_pt_counter += 1
                # early stopping check
                if es_pt_counter >= config['es_patience']:
                    print('Stopping training early since no loss improvement over {} epochs.'
                          .format(str(es_pt_counter)))
                    early_stop = True
                    # print results of best epoch
                    print('Final (best) results: ')
                    print("Train Loss: {:.4f}".format(np.mean(best_train_losses)),
                          "Train Acc: {:.4f}".format(jaccard_score(train_gt, best_train_preds, average='macro', labels=labels)),
                          "Train Prec: {:.4f}".format(precision_score(train_gt, best_train_preds, average='macro', labels=labels)),
                          "Train Rcll: {:.4f}".format(recall_score(train_gt, best_train_preds, average='macro', labels=labels)),
                          "Train F1: {:.4f}".format(f1_score(train_gt, best_train_preds, average='macro', labels=labels)),
                          "Val Loss: {:.4f}".format(np.mean(best_val_losses)),
                          "Val Acc: {:.4f}".format(jaccard_score(val_gt, best_val_preds, average='macro', labels=labels)),
                          "Val Prec: {:.4f}".format(precision_score(val_gt, best_val_preds, average='macro', labels=labels)),
                          "Val Rcll: {:.4f}".format(recall_score(val_gt, best_val_preds, average='macro', labels=labels)),
                          "Val F1: {:.4f}".format(f1_score(val_gt, best_val_preds, average='macro', labels=labels)))
        else:
            print(f"Performance improved... ({best_metric}->{metric})")
            if config['early_stopping']:
                es_pt_counter = 0
                best_train_losses = train_losses
                best_val_losses = val_losses
            best_metric = metric
            best_network = network
            checkpoint = {
                "model_state_dict": network.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "random_rnd_state": random.getstate(),
                "numpy_rnd_state": np.random.get_state(),
                "torch_rnd_state": torch.get_rng_state(),
            }
            best_train_preds = train_preds
            best_val_preds = val_preds

        # set network to train mode again
        network.train()

        if early_stop:
            break

    # if plot_gradient gradient plot is shown at end of training
    if config['save_gradient_plot']:
        if config['name']:
            plt.savefig(os.path.join(log_dir, 'grad_flow_{}.png'.format(config['name'])))
        else:
            plt.savefig(os.path.join(log_dir, 'grad_flow.png'))

    # return validation, train and test predictions as numpy array with ground truth
    if config['valid_epoch'] == 'best':
        return best_network, checkpoint, np.vstack((best_val_preds, val_gt)).T, \
               np.vstack((best_train_preds, train_gt)).T
    else:
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }
        return network, checkpoint, np.vstack((val_preds, val_gt)).T, np.vstack((train_preds, train_gt)).T


def train_regression(train_features, train_labels, val_features, val_labels, network, optimizer,
         loss, lr_scheduler=None, log_dir=None, y_scaler=None):
    """
    Method to train a PyTorch network.

    :param train_features: numpy array
        Training features
    :param train_labels: numpy array
        Training labels
    :param val_features: numpy array
        Validation features
    :param val_labels: numpy array
        Validation labels
    :param network: pytorch model
        DeepConvLSTM network object
    :param optimizer: optimizer object
        Optimizer object
    :param loss: loss object
        Loss object
    :param config: dict
        Config file which contains all training and hyperparameter settings
    :param log_date: string
        Date used for logging
    :param log_timestamp: string
        Timestamp used for logging
    :param lr_scheduler: scheduler object, default: None
        Learning rate scheduler object
    :return pytorch model, numpy array, numpy array
        Trained network and training and validation predictions with ground truth
    """

    # prints the number of learnable parameters in the network
    count_parameters(network)

    # init network using weight initialization of choice
    network = init_weights(network)
    # send network to GPU
    network.to(config['gpu'])
    network.train()


    # initialize optimizer and loss
    opt, criterion = optimizer, loss

    # if config['loss'] == 'maxup':
    #     maxup = Maxup(my_noise_addition_augmenter, ntrials=4)

    # initialize training and validation dataset, define DataLoaders
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))

    g = torch.Generator()
    g.manual_seed(config['seed'])

    trainloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffling'],
                             worker_init_fn=seed_worker, generator=g, pin_memory=True)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    valloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                           worker_init_fn=seed_worker, generator=g, pin_memory=True)

    # counters and objects used for early stopping and learning rate adjustment
    best_metric = float("inf")
    best_network = None
    val_losses = None
    train_losses_epoch = []
    val_losses_epoch = []
    best_train_preds = None
    early_stop = False
    es_pt_counter = 0
    labels = list(range(0, config['nb_classes']))

    # training loop; iterates through epochs
    for e in range(config['epochs']):
        """
        TRAINING
        """
        # helper objects
        train_preds = []
        train_gt = []
        train_losses = []
        start_time = time.time()
        batch_num = 0

        # iterate over train dataset
        for i, (x, y) in enumerate(trainloader):
            # send x and y to GPU
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
            # zero accumulated gradients
            opt.zero_grad()

            train_output = network(inputs)
 
            train_loss = criterion(train_output, targets)

            train_loss.backward()
            opt.step()
            # append train loss to list
            train_losses.append(train_loss.item())

            # create predictions and append them to final list
            y_preds = train_output.cpu().detach().numpy().flatten()
            y_true = targets.cpu().numpy().flatten()
            train_preds = np.concatenate((np.array(train_preds, float), np.array(y_preds, float)))
            train_gt = np.concatenate((np.array(train_gt, float), np.array(y_true, float)))
            train_preds_un = y_scaler.inverse_transform(train_preds.reshape(-1,1))
            train_gt_un = y_scaler.inverse_transform(train_gt.reshape(-1,1))
            # if verbose print out batch wise results (batch number, loss and time)
            if batch_num % config['print_freq'] == 0 and batch_num > 0:
                cur_loss = np.mean(train_losses)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d} batches | ms/batch {:5.5f} | '
                        'train loss {:5.5f}'.format(e, batch_num, elapsed * 1000 / config['batch_size'], cur_loss))
                start_time = time.time()
            batch_num += 1
            # plot gradient flow if wanted
            if config['save_gradient_plot']:
                plot_grad_flow(network)
        
        train_losses_epoch.append(np.mean(train_losses))  # save the epoch mean of train_loss
        """
        VALIDATION
        """

        # helper objects
        val_preds = []
        val_gt = []
        val_losses = []
        fp = np.zeros((config['nb_classes'],1))  #false positives
        fn = np.zeros((config['nb_classes'],1)) #false negatives
        mse = np.zeros((config['nb_classes'],1))
        precision = np.zeros((config['nb_classes'],1)) # given the margin of error in grams
        elementCounter = np.zeros((config['nb_classes'],1))
        # set network to eval mode
        network.eval()
        with torch.no_grad():
            # iterate over validation dataset
            for i, (x, y) in enumerate(valloader):
                # send x and y to GPU
                inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

                # if config['loss'] == 'maxup':
                #     # Increase the inputs via data augmentation
                #     inputs, targets = maxup(inputs, targets)

                # send inputs through network to get predictions, loss and calculate softmax probabilities
                val_output = network(inputs)

                val_loss = criterion(val_output, targets)

                val_losses.append(val_loss.item())

                # create predictions and append them to final list
                y_preds = val_output.cpu().numpy().flatten()
                y_true = targets.cpu().numpy().flatten()
                val_preds = np.concatenate((np.array(val_preds, float), np.array(y_preds, float)))
                val_gt = np.concatenate((np.array(val_gt, float), np.array(y_true, float)))
                
        val_losses_epoch.append(np.mean(val_losses)) # Calculate the loss in the validation set per the 10 epochs
        #print("validation loss: ", cur_val_loss)
        # employ early stopping if employed
        # metric = f1_score(val_gt, val_preds, average='macro')
        metric_scaled = mean_squared_error(val_gt, val_preds)
        metric_scaled = np.sqrt(metric_scaled)
        val_gt_un = y_scaler.inverse_transform(val_gt.reshape(-1,1))
        val_preds_un = y_scaler.inverse_transform(val_preds.reshape(-1,1))
        # print("Real Values:", val_gt_un)
        # print("Predicted values: ", val_preds_un)metric_scaled
        metric_unscaled = mean_squared_error(val_gt_un, val_preds_un, squared=False)

        #calculating the FPs, FNs and precision with +-5gr
        for i in range(len(val_gt_un)):
            y_trueVal = round(val_gt_un[i,0])
            if y_trueVal == 16:
                pos = 0
            elif y_trueVal == 23:
                pos = 1
            elif y_trueVal == 38:
                pos = 2
            elif y_trueVal == 66:
                pos = 3
            elif y_trueVal == 74:
                pos = 4
            elif y_trueVal ==303:
                pos = 5
            elif y_trueVal == 595:
                pos = 6
            else:
                raise ValueError("y_trueVal did not match the available labels")
            
            #check if prediction matches the true value
            if ((val_preds_un[i,0] > (y_trueVal - config['error_margins'])) and (val_preds_un[i,0] < (y_trueVal + config['error_margins']))):
                precision[pos,0] = precision[pos,0] + 1
            elif (val_preds_un[i] > (y_trueVal + config['error_margins'])):
                fp[pos,0] = fp[pos,0] + 1
            elif (val_preds_un[i] < (y_trueVal - config['error_margins'])):
                fn[pos, 0] = fn[pos, 0] + 1
            #count the mse and increase the counter
            mse[pos,0] = mse[pos,0] + (val_gt_un[i,0] - val_preds_un[i,0])**2
            elementCounter[pos,0] = elementCounter[pos,0] + 1 

        precision = precision / elementCounter
        print("Precision per label/weight")
        print(precision.T)
        print("false positives per label/weight")
        print(fp.T)
        print("false negatives per label/weight")
        print(fn.T)
        print("RMSE per label/weight")
        rmse = np.sqrt(mse / elementCounter)
        print(rmse.T)
        print("Counter")
        print(elementCounter.T)

        # metric_unscaled = np.sqrt(metric_unscaled)
        print("metric_unscaled", metric_unscaled)
        if metric_unscaled < best_metric:
            print(f"RMSE improved... ({best_metric}->{metric_unscaled})")
            best_metric = metric_unscaled
            best_network = network
            checkpoint = {
                "model_state_dict": network.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "random_rnd_state": random.getstate(),
                "numpy_rnd_state": np.random.get_state(),
                "torch_rnd_state": torch.get_rng_state(),
            }
            best_train_preds = train_preds_un
            best_val_preds = val_preds_un
            best_fp = fp.flatten().tolist()
            best_fn = fn.flatten().tolist()
            best_precision = rmse.flatten().tolist()
            best_elementCounter = elementCounter.flatten().tolist()

        # set network to train mode again
        network.train()

        if early_stop:
            break

    # if plot_gradient gradient plot is shown at end of training
    if config['save_gradient_plot']:
        if config['name']:
            plt.savefig(os.path.join(log_dir, 'grad_flow_{}.png'.format(config['name'])))
        else:
            plt.savefig(os.path.join(log_dir, 'grad_flow.png'))

    # return validation, train and test predictions as numpy array with ground truth
    if config['valid_epoch'] == 'best':
        return best_network, checkpoint, np.vstack((best_val_preds, val_gt_un)).T, \
               np.vstack((best_train_preds, train_gt_un)).T, best_fp, best_fn, best_precision, \
            best_elementCounter, train_losses_epoch, val_losses_epoch
    else:   
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }
        return network, checkpoint, np.vstack((val_preds, val_gt)).T, np.vstack((train_preds, train_gt)).T
