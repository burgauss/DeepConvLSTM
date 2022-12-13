import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

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
                    network, optimizer, loss, config, log_date, log_timestamp):
    
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