import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data

    # EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load("features.npz", allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)

    def KFold_cross_validation(model_class, X, Y, K, param_name, param_value, **kwargs):
        """
        Generic K-Fold cross-validation.

        Arguments:
            model_class: the class of the model to use (e.g., KNN or LogisticRegression)
            X: training data
            Y: training labels
            K: number of folds
            param_name: name of the hyperparameter to tune (e.g., 'k' or 'lr')
            param_value: the value to test for that hyperparameter
            kwargs: any other keyword args to pass (e.g., max_iters)

        Returns:
            Average validation accuracy
        """
        N = X.shape[0]
        fold_size = N // K
        indices = np.random.permutation(N)
        accuracies = []

        for fold in range(K):
            val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]

            # Construct model dynamically
            model = model_class(**{param_name: param_value}, **kwargs)
            model.fit(X_train, Y_train)
            preds = model.predict(X_val)
            acc = accuracy_fn(preds, Y_val)
            accuracies.append(acc)

        return np.mean(accuracies)


    def run_cv_for_hyperparam(model_class, X, Y, K, param_name, param_list, **kwargs):
        """
        Runs cross-validation for multiple values of a hyperparameter.

        Returns:
            List of average accuracies.
        """
        model_performance = []

        for val in param_list:
            print(f"Testing {param_name}={val}")
            acc = KFold_cross_validation(model_class, X, Y, K, param_name, val, **kwargs)
            model_performance.append(acc)

        return model_performance
    if not args.test:
        if args.method == "kmeans":
            split_idx = int(xtrain.shape[0] * 0.8)
            xval = xtrain[split_idx:]
            yval = ytrain[split_idx:]
            xtrain = xtrain[:split_idx]
            ytrain = ytrain[:split_idx]
            xtest, ytest = xval, yval

        elif args.method == "knn":
            K = 4
            k_list = range(1, 7)
            print("Running cross-validation for kNN...")
            model_performance = run_cv_for_hyperparam(KNN, xtrain, ytrain, K, "k", k_list)

            best_k = k_list[np.argmax(model_performance)]
            print(f"Best k: {best_k}")
            method_obj = KNN(k=best_k)

        elif args.method == "logistic_regression":
            K = 4
            lr_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            print("Running cross-validation for Logistic Regression...")
            model_performance = run_cv_for_hyperparam(LogisticRegression, xtrain, ytrain, K, "lr", lr_list, max_iters=args.max_iters)

            best_lr = lr_list[np.argmax(model_performance)]
            print(f"Best learning rate: {best_lr}")
            method_obj = LogisticRegression(lr=best_lr, max_iters=args.max_iters)


    ### WRITE YOUR CODE HERE to do any other data processing
    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)

    ## 3. Initialize the method you want to use.
    
    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K)
    else:
        raise ValueError(f"Unknown method: {args.method}. Please specify a valid method.")    

    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data for classification task
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### Visualization 
    if not args.test:
        if args.method == "knn":
            plt.plot(k_list, model_performance)
            plt.title("Validation accuracy vs. k (kNN)")
            plt.xlabel("k")
            plt.ylabel("Accuracy")
            plt.show()

        elif args.method == "logistic_regression":
            plt.plot(lr_list, model_performance)
            plt.xscale("log")
            plt.title("Validation accuracy vs. learning rate (Logistic Regression)")
            plt.xlabel("Learning rate")
            plt.ylabel("Accuracy")
            plt.show()

        elif args.method == "kmeans":
            print("Running elbow method...")
            wcss = method_obj.average_wcss(xtrain, ytrain, max_k=10, n_init=10)
            plt.plot(range(1, 11), wcss)
            plt.title("Elbow Method (AWCSS vs. k)")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Average WCSS")
            plt.show()
    
    
if __name__ == "__main__":
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)",
    )
    parser.add_argument(
        "--data_path", default="data", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="features/original(MS2)"
    )
    parser.add_argument(
        "--K", type=int, default=1, help="number of neighboring datapoints used for knn and kmeans"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument(
        "--nn_type",
        default="cnn",
        help="which network to use, can be 'Transformer' or 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
