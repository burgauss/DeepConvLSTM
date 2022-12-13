from model.models import DeepConvLSTM_Simplified
from config import config
import torch
from torch.utils.data import DataLoader
import numpy as np

def validation_simplified(modelName, config,  val_features, val_labels, scaler):
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