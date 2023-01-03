from model.models import DeepConvLSTM_Simplified, DeepConvLSTM, DeepConvLSTM_regression
from config import config
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, mean_squared_error
from model.train import train, train_regression

def validation_simplified(modelName, val_features, val_labels, scaler):
    """Function gets a saved model and takes one batch to make predictions just for visualization purposes
    params 
        modelName: String
            a string with the name of the model that is going to be uploaded
        val_features : np.array
            a normalized validation dataset. see example above. X_train_ss.astype(np.float32)
        val_labels: np.array
            a normalized or unnormalized y_val label dataset configured to work in the gpu.y_train.astype(np.uint8)
    """
    #upload the model
    net = DeepConvLSTM_Simplified(config=config)
    # net.load_state_dict(torch.load('model2.pth'))
    net.load_state_dict(torch.load(modelName))
    net.to(config['gpu'])
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    valLoader = DataLoader(valid_dataset, batch_size = config['batch_size'], shuffle=False)

    for i, (x,y) in enumerate(valLoader):
        samples_x = x
        samples_y = y
        samples_x_np = samples_x.numpy()
        samples_y_np = samples_y.numpy()
        print("Shape X: ", samples_x.shape)
        print("Shape y: ", samples_y.shape)
        if i == 1:
            break
    batches = len(samples_x_np)
    print("Example of one sample data: ", samples_x_np[5,:,0])
    print("Label: ", samples_y_np[5])

    #Unscaling the data using the scaler object
    samples_x_np_flat = samples_x_np.flatten().reshape(-1,1)
    sample_unnorm_np = scaler.inverse_transform(samples_x_np_flat)
    sample_unnorm_np = sample_unnorm_np.reshape(batches, -1, 1)

    print("unscaled values: ", sample_unnorm_np[5,:,0])
    print("Average of the unscaled values, ", np.mean(sample_unnorm_np[5,:,0]))

    #Getting the predictions
    net.eval()
    with torch.no_grad():
        inputs, targets = samples_x.to(config['gpu']), samples_y.to(config['gpu'])

        val_output = net(inputs)    #forwards pass

        val_output = torch.nn.functional.softmax(val_output, dim=1)

        y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
        y_true = targets.cpu().numpy().flatten()

    print("predicted labels: ", y_preds)
    print("true labels: ", y_true)

def validation(modelName, val_features, val_labels, scaler):
    """Function gets a saved model and takes one batch to make predictions just for visualization purposes
    params 
        modelName: String
            a string with the name of the model that is going to be uploaded
        val_features : np.array
            a normalized validation dataset. see example above. X_train_ss.astype(np.float32)
        val_labels: np.array
            a normalized or unnormalized y_val label dataset configured to work in the gpu.y_train.astype(np.uint8)
    """
    #upload the model
    net = DeepConvLSTM(config=config)
    # net.load_state_dict(torch.load('model2.pth'))
    net.load_state_dict(torch.load(modelName))
    net.to(config['gpu'])
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    valLoader = DataLoader(valid_dataset, batch_size = config['batch_size'], shuffle=False)

    for i, (x,y) in enumerate(valLoader):
        samples_x = x
        samples_y = y
        samples_x_np = samples_x.numpy()
        samples_y_np = samples_y.numpy()
        print("Shape X: ", samples_x.shape)
        print("Shape y: ", samples_y.shape)
        if i == 1:
            break
    batches = len(samples_x_np)
    print("Example of one sample data: ", samples_x_np[5,:,0])
    print("Label: ", samples_y_np[5])

    #Unscaling the data using the scaler object
    samples_x_np_flat = samples_x_np.flatten().reshape(-1,1)
    sample_unnorm_np = scaler.inverse_transform(samples_x_np_flat)
    sample_unnorm_np = sample_unnorm_np.reshape(batches, -1, 1)

    print("unscaled values: ", sample_unnorm_np[5,:,0])
    print("Average of the unscaled values, ", np.mean(sample_unnorm_np[5,:,0]))

    #Getting the predictions
    net.eval()
    with torch.no_grad():
        inputs, targets = samples_x.to(config['gpu']), samples_y.to(config['gpu'])

        val_output = net(inputs)    #forwards pass

        val_output = torch.nn.functional.softmax(val_output, dim=1)

        y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
        y_true = targets.cpu().numpy().flatten()

    print("predicted labels: ", y_preds)
    print("true labels: ", y_true)

def validation_regression(modelName, val_features, val_labels, x_scaler, y_scaler):
    """Function gets a saved model and takes one batch to make predictions just for visualization purposes
    params, for the specific case of a regression
        modelName: String
            a string with the name of the model that is going to be uploaded
        val_features : np.array
            a normalized validation dataset. see example above. X_train_ss.astype(np.float32)
        val_labels: np.array
            a normalized or unnormalized y_val label dataset configured to work in the gpu.y_train.astype(np.uint8)
    """
    #upload the model
    net = DeepConvLSTM_regression(config=config)
    # net.load_state_dict(torch.load('model2.pth'))
    net.load_state_dict(torch.load(modelName))
    net.to(config['gpu'])
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    valLoader = DataLoader(valid_dataset, batch_size = config['batch_size'], shuffle=False)

    for i, (x,y) in enumerate(valLoader):
        samples_x = x
        samples_y = y
        samples_x_np = samples_x.numpy()
        samples_y_np = samples_y.numpy()
        if i >= 3:
            break

    print("Shape X: ", samples_x.shape)
    print("Shape y: ", samples_y.shape)

    batches = len(samples_x_np)
    print("Example of one sample data: ", samples_x_np[5,:,0])
    print("scaled Label: ", samples_y_np[5])

    #Unscaling the data using the scaler object
    samples_x_np_flat = samples_x_np.flatten().reshape(-1,1)
    sample_unnorm_np = x_scaler.inverse_transform(samples_x_np_flat)
    sample_unnorm_np = sample_unnorm_np.reshape(batches, -1, 1)

    label_unnorm_np = y_scaler.inverse_transform(samples_y_np)
    label_unnorm_np = label_unnorm_np.reshape(batches, -1, 1)

    print("unscaled values: ", sample_unnorm_np[5,:,0])
    print("unscaled label: ", label_unnorm_np[5])
    print("Average of the unscaled values, ", np.mean(sample_unnorm_np[5,:,0]))

    #Getting the predictions
    net.eval()
    with torch.no_grad():
        inputs, targets = samples_x.to(config['gpu']), samples_y.to(config['gpu'])

        val_output = net(inputs)    #forwards pass

        # val_output = torch.nn.functional.softmax(val_output, dim=1)

        # y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
        y_preds = val_output.cpu().numpy()
        y_true = targets.cpu().numpy()
    # unscaling
    y_preds_unscaled = y_scaler.inverse_transform(y_preds)
    y_true_unscaled = y_scaler.inverse_transform(y_true)    

    print("predicted labels: ", y_preds_unscaled)
    print("true labels: ", y_true_unscaled)

def validation_regressionComplete(modelName, val_features, val_labels, x_scaler, y_scaler):
    """Function gets a saved model and takes one batch to make predictions just for visualization purposes
    params, for the specific case of a regression
        modelName: String
            a string with the name of the model that is going to be uploaded
        val_features : np.array
            a normalized validation dataset. see example above. X_train_ss.astype(np.float32)
        val_labels: np.array
            a normalized or unnormalized y_val label dataset configured to work in the gpu.y_train.astype(np.uint8)
    """
    #upload the model
    net = DeepConvLSTM_regression(config=config)
    # net.load_state_dict(torch.load('model2.pth'))
    net.load_state_dict(torch.load(modelName))
    net.to(config['gpu'])
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    valLoader = DataLoader(valid_dataset, batch_size = config['batch_size'], shuffle=False)

    val_preds = []
    val_gt = []
    fp = np.zeros((config['nb_classes'],1))  #false positives
    fn = np.zeros((config['nb_classes'],1)) #false negatives
    mse = np.zeros((config['nb_classes'],1))
    precision = np.zeros((config['nb_classes'],1)) # given the margin of error in grams
    elementCounter = np.zeros((config['nb_classes'],1))

    # for i, (x,y) in enumerate(valLoader):
    #     samples_x = x
    #     samples_y = y
    #     samples_x_np = samples_x.numpy()
    #     samples_y_np = samples_y.numpy()
    #     if i >= 3:
    #         break


    #Getting the predictions
    net.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(valLoader):

            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

            val_output = net(inputs)    #forwards pass

            # val_output = torch.nn.functional.softmax(val_output, dim=1)

            # y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
            y_preds = val_output.cpu().numpy().flatten()
            y_true = targets.cpu().numpy().flatten()
            # unscaling
            # y_preds_unscaled = y_scaler.inverse_transform(y_preds)
            # y_true_unscaled = y_scaler.inverse_transform(y_true)    
            val_preds = np.concatenate((np.array(val_preds, float), np.array(y_preds, float)))
            val_gt = np.concatenate((np.array(val_gt, float), np.array(y_true, float)))
    
        val_gt_un = y_scaler.inverse_transform(val_gt.reshape(-1,1))
        val_preds_un = y_scaler.inverse_transform(val_preds.reshape(-1,1))
        # print("Real Values:", val_gt_un)
        # print("Predicted values: ", val_preds_un)metric_scaled
        metric_unscaled = mean_squared_error(val_gt_un, val_preds_un, squared=False)  #No needed, we have the manual way of calculation

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
        print("Overall validation dataset RMSE:")
        print(metric_unscaled)
        print("false positives per label/weight")
        print(fp.T)
        print("false negatives per label/weight")
        print(fn.T)
        print("RMSE per label/weight")
        rmse = np.sqrt(mse / elementCounter)
        print(rmse.T)
        print("Counter")
        print(elementCounter.T)


def train_valid_split(x_train_set, y_train_set,
             x_valid_set, y_valid_set, custom_net, custom_loss, custom_opt, log_dir=None, y_scaler=None):
    """
    Method to apply normal cross-validation, i.e. one set split into train, validation and testing data.

    :param x_train_data: numpy array
        Data used for training
    :param y_train_set: numpy array
        labels of the train dataset
    :param x_valid_set: numpy array
        Data used for validation
    :param y_valid_set: numpy array
        labels of the label dataset
    :param custom_net: pytorch model
        Custom network object
    :param custom_loss: loss object
        Custom loss object
    :param custom_opt: optimizer object
        Custom optimizer object
    :param log_dir: string
        Logging directory
    :return pytorch model
        Trained network
    """
    print('\nCALCULATING TRAIN-VALID-SPLIT SCORES.\n')
    # Sensor data is segmented using a sliding window mechanism
    X_train = x_train_set
    y_train = y_train_set
    X_val = x_valid_set
    y_val = y_valid_set

    # network initialization
    net = custom_net

    # optimizer initialization
    opt = custom_opt

    # optimizer initialization
    loss = custom_loss

    # # lr scheduler initialization
    # if config['adj_lr']:           
    #     print('Adjusting learning rate according to scheduler: ' + args.lr_scheduler)
    #     scheduler = init_scheduler(opt, args)
    # else:
    #     scheduler = None
    if config['DL_mode'] == 'classification':
        net, checkpoint, val_output, train_output = train(X_train, y_train, X_val, y_val,
                                                      network=net, optimizer=opt, loss=loss, lr_scheduler=None,
                                                      log_dir=log_dir)
        
        labels = list(range(0, config['nb_classes']))
        train_acc = jaccard_score(train_output[:, 1], train_output[:, 0], average=None, labels=labels)
        train_prec = precision_score(train_output[:, 1], train_output[:, 0], average=None, labels=labels)
        train_rcll = recall_score(train_output[:, 1], train_output[:, 0], average=None, labels=labels)
        train_f1 = f1_score(train_output[:, 1], train_output[:, 0], average=None, labels=labels)

        val_acc = jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        val_prec = precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        val_rcll = recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        val_f1 = f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)

        print('VALIDATION RESULTS (macro): ')
        print("Avg. Accuracy: {0}".format(np.average(val_acc)))
        print("Avg. Precision: {0}".format(np.average(val_prec)))
        print("Avg. Recall: {0}".format(np.average(val_rcll)))
        print("Avg. F1: {0}".format(np.average(val_f1)))

        print("VALIDATION RESULTS (PER CLASS): ")
        print("Accuracy: {0}".format(val_acc))
        print("Precision: {0}".format(val_prec))
        print("Recall: {0}".format(val_rcll))
        print("F1: {0}".format(val_f1))

        print("GENERALIZATION GAP ANALYSIS: ")
        print("Train-Val-Accuracy Difference: {0}".format(np.average(train_acc) - np.average(val_acc)))
        print("Train-Val-Precision Difference: {0}".format(np.average(train_prec) - np.average(val_prec)))
        print("Train-Val-Recall Difference: {0}".format(np.average(train_rcll) - np.average(val_rcll)))
        print("Train-Val-F1 Difference: {0}".format(np.average(train_f1) - np.average(val_f1)))
    
    elif config['DL_mode'] == 'regression':
        net, checkpoint, val_output, train_output, best_fp, best_fn, best_precision, counter = train_regression(X_train, y_train, X_val, y_val,
                                                network=net, optimizer=opt, loss=loss, lr_scheduler=None,
                                                log_dir=log_dir, y_scaler=y_scaler)                             





    return net, checkpoint, val_output, train_output, best_fp, best_fn, best_precision, counter 