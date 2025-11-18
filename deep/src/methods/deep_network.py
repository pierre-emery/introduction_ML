import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, dropout= 0.0):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = F.relu(self.fc3(x)); x = self.dropout(x)
        x = F.relu(self.fc4(x)); x = self.dropout(x)
        return self.fc5(x)


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, dropout=0.0):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32,           64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64,          128, kernel_size=5, padding=2)
        self.dropout_conv = nn.Dropout2d(dropout)
        self.dropout_fc   = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x,2); x = self.dropout_conv(x)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x,2); x = self.dropout_conv(x)
        x = F.relu(self.conv3(x)); x = F.max_pool2d(x,2); x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout_fc(x)
        preds = self.fc2(x)
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        print(f"Training epoch {ep + 1}/{self.epochs}")
        self.model.train()
        for x, y in dataloader:
            preds = self.model(x)
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                preds = self.model(x)
                labels = torch.argmax(preds, dim=1)
                all_preds.append(labels)
        pred_labels = torch.cat(all_preds)
        return pred_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        assert training_data.shape[0] == training_labels.shape[0], "Mismatch between X and Y"
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        return pred_labels.cpu().numpy()

class CrossValidTrainer:
    """
    Same function as the Trainer Class but this one will be used when we run cross validation.

    """
    def __init__(self, hyperparam_value, device='cpu', nn_type='mlp', epochs=100, batch_size=64,
             dropout=0.0, lr=None, **kwargs):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

        if lr is None:
            lr_val = hyperparam_value
        else:
            lr_val = lr

        if dropout is None:
            dropout_val = hyperparam_value
        else:
            dropout_val = dropout

        if batch_size is None:
            batch_size_val = hyperparam_value
        else:
            batch_size_val = batch_size

        self.batch_size = batch_size_val

        print(f"Initializing CrossValidTrainer with lr={lr_val}, dropout={dropout_val}, batch_size={self.batch_size}")

        if nn_type == 'mlp':
            self.model = MLP(input_size=28*28*3, n_classes=7, dropout=dropout_val)
        elif nn_type == 'cnn':
            self.model = CNN(input_channels=3, n_classes=7, dropout=dropout_val)
        else:
            raise ValueError(f"Unknown nn_type {nn_type}")

        self.model.to(device)

        self.trainer = Trainer(self.model, lr=lr_val, epochs=epochs, batch_size=self.batch_size)

    def fit(self, X_train, Y_train):
        return self.trainer.fit(X_train, Y_train)

    def predict(self, X_val):
        return self.trainer.predict(X_val)