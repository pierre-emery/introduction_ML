import numpy as np 


# Generaly utilies
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)

def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (array): of shape (N,D)
    Returns:
        (array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]),data], axis=1)
    return data

def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds

def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.

def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        macrof1 += 2*(precision*recall)/(precision+recall)

    return macrof1/len(class_ids)

def mse_fn(pred,gt):
    '''
        Mean Squared Error
        Arguments:
            pred: NxD prediction matrix
            gt: NxD groundtruth values for each predictions
        Returns:
            returns the computed loss

    '''
    loss = (pred-gt)**2
    loss = np.mean(loss)
    return loss

#Functions we added to run k-fold cross validation 
def KFold_cross_validation(model_class, X, Y, K, param_name, param_value, device="cpu", nn_type="mlp", **kwargs):
    N = X.shape[0]
    fold_size = N // K
    indices = np.random.permutation(N)
    accuracies = []

    for fold in range(K):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        if nn_type == "mlp":
            X_train_ = X_train.reshape(X_train.shape[0], -1)
            X_val_ = X_val.reshape(X_val.shape[0], -1)
        elif nn_type == "cnn":
            X_train_ = np.transpose(X_train, (0, 3, 1, 2))
            X_val_ = np.transpose(X_val, (0, 3, 1, 2))
        else:
            raise ValueError(f"Unknown nn_type {nn_type}")

        trainer = model_class(param_value, device=device, nn_type=nn_type, **kwargs)

        trainer.fit(X_train_, Y_train)
        preds = trainer.predict(X_val_)
        acc = accuracy_fn(preds, Y_val)
        accuracies.append(acc)

    return np.mean(accuracies)

def run_cv_for_hyperparam(model_class, X, Y, K, param_name, param_list, device="cpu", nn_type="mlp", **kwargs):
    performances = []
    for val in param_list:
        print(f"Testing {param_name}={val}")
        perf = KFold_cross_validation(model_class, X, Y, K, param_name, val, device=device, nn_type=nn_type, **kwargs)
        performances.append(perf)
    return performances
    