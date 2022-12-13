from model.models import DeepConvLSTM_Simplified
from config import config
import torch
from torch.utils.data import DataLoader
import numpy as np
from model.train import train

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


def train_valid_split(x_train_set, y_train_set,
             x_valid_set, y_valid_set, custom_net, custom_loss, custom_opt, log_dir=None):
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

    net, checkpoint, val_output, train_output = train(X_train, y_train, X_val, y_val,
                                                      network=net, optimizer=opt, loss=loss, lr_scheduler=None,
                                                      log_dir=log_dir)
                                                      

    if args.save_checkpoints:
        print('Saving checkpoint...')
        if args.valid_epoch == 'last':
            if args.name:
                c_name = os.path.join(log_dir, "checkpoint_last_{}.pth".format(str(args.name)))
            else:
                c_name = os.path.join(log_dir, "checkpoint_last.pth")
        else:
            if args.name:
                c_name = os.path.join(log_dir, "checkpoint_best_{}.pth".format(str(args.name)))
            else:
                c_name = os.path.join(log_dir, "checkpoint_best.pth")
        torch.save(checkpoint, c_name)

    labels = list(range(0, args.nb_classes))
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

    if args.save_analysis:
        tv_results = pd.DataFrame([val_acc, val_prec, val_rcll, val_f1], columns=args.class_names)
        tv_results.index = ['accuracy', 'precision', 'recall', 'f1']
        tv_gap = pd.DataFrame([train_acc - val_acc, train_prec - val_prec, train_rcll - val_rcll, train_f1 - val_f1],
                              columns=args.class_names)
        tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        if args.name:
            tv_results.to_csv(os.path.join(log_dir, 'split_scores_{}.csv'.format(args.name)))
            tv_gap.to_csv(os.path.join(log_dir, 'tv_gap_{}.csv'.format(args.name)))
        else:
            tv_results.to_csv(os.path.join(log_dir, 'split_scores.csv'))
            tv_gap.to_csv(os.path.join(log_dir, 'tv_gap.csv'))

    evaluate_split_scores(input_cm=val_output,
                          class_names=args.class_names,
                          filepath=log_dir,
                          filename='split',
                          args=args
                          )
    return net