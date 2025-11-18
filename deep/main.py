import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer, CrossValidTrainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, KFold_cross_validation, run_cv_for_hyperparam
import matplotlib.pyplot as plt

import time


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                        of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data()

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        split = int(0.8 * len(xtrain))
        xval = xtrain[split:]
        yval = ytrain[split:]
        xtrain = xtrain[:split]
        ytrain = ytrain[:split]
    else:
        xval, yval = xtest, ytest

    ### WRITE YOUR CODE HERE to do any other data processing
    mean = np.mean(xtrain, axis=0)
    std = np.std(xtrain, axis=0) + 1e-8  
    xtrain = normalize_fn(xtrain, mean, std)
    xval = normalize_fn(xval, mean, std)

    cv_results = None    # We keep the values in these two lists for the visualization later.
    cv_param_list = None

    if args.cv == 'lr':
        cv_param_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        print("Running cross-validation for learning rate...")
        performances = run_cv_for_hyperparam(
            CrossValidTrainer, xtrain, ytrain, K=4, param_name='lr', param_list=cv_param_list,
            device=args.device, nn_type=args.nn_type, epochs=args.max_iters,
            batch_size=args.nn_batch_size, dropout=args.dropout, lr=None
        )
        best_val = cv_param_list[np.argmax(performances)]
        print(f"Best learning rate found: {best_val}")
        args.lr = best_val
        cv_results = performances

    elif args.cv == 'dropout':
        cv_param_list = [0.0, 0.1, 0.3, 0.5]
        print("Running cross-validation for dropout rate...")
        performances = run_cv_for_hyperparam(
            CrossValidTrainer, xtrain, ytrain, K=4, param_name='dropout', param_list=cv_param_list,
            device=args.device, nn_type=args.nn_type, epochs=args.max_iters,
            batch_size=args.nn_batch_size, lr=args.lr, dropout=None
        )
        best_val = cv_param_list[np.argmax(performances)]
        print(f"Best dropout rate found: {best_val}")
        args.dropout = best_val
        cv_results = performances

    elif args.cv == 'bs':
        cv_param_list = [16, 32, 64, 128]
        print("Running cross-validation for batch size...")
        performances = run_cv_for_hyperparam(
            CrossValidTrainer, xtrain, ytrain, K=5, param_name='batch_size', param_list=cv_param_list,
            device=args.device, nn_type=args.nn_type, epochs=args.max_iters,
            dropout=args.dropout, lr=args.lr, batch_size=None
        )
        best_val = cv_param_list[np.argmax(performances)]
        print(f"Best batch size found: {best_val}")
        args.nn_batch_size = best_val
        cv_results = performances

### To complicated, we didn't implement it in the end 
    """elif args.cv == 'hidden_dims':
        if args.nn_type == 'mlp':
            cv_param_list = [
                [256, 128, 64],
                [512, 256, 128],
                [1024, 512, 256],
            ]
        elif args.nn_type == 'cnn':
            cv_param_list = [
                (8, 16, 32),
                (16, 32, 64),
                (32, 64, 128),
            ]
        else:
            raise ValueError(f"hidden_dims CV only supported for mlp|cnn, got {args.nn_type}")

        print(f"Running cross-validation for hidden-layer dimensions on {args.nn_type}…")
        param_list = [tuple(d) for d in cv_param_list]

        performances = run_cv_for_hyperparam(CrossValidTrainer, xtrain, ytrain, K=5, param_name='hidden_dims',
            param_list=param_list, device=args.device, nn_type=args.nn_type, epochs=args.max_iters, batch_size=args.nn_batch_size,
            lr=args.lr, dropout=args.dropout, hidden_dims=None, filters=None
        )

        best_idx = np.argmax(performances)
        best_dims = cv_param_list[best_idx]
        print(f"Best hidden-layer dims found: {best_dims}")
        args.hidden_dims = best_dims
        cv_results = performances"""


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xval = xval.reshape(xval.shape[0], -1)
        model = MLP(input_size=xtrain.shape[1], n_classes=get_n_classes(ytrain), dropout = args.dropout)
    elif args.nn_type == "cnn":
        xtrain = np.transpose(xtrain, (0, 3, 1, 2))
        xval = np.transpose(xval, (0, 3, 1, 2))
        model = CNN(input_channels=3, n_classes=n_classes, dropout = args.dropout)
    else:
        raise ValueError(f"Unknown NN type: {args.nn_type}")

    print(summary(model, input_size=(args.nn_batch_size,) + tuple(xtrain.shape[1:])))

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    print("xtrain shape:", xtrain.shape)
    print("ytrain shape:", ytrain.shape)
    print("Starting training…")
    t0 = time.time()
    preds_train = method_obj.fit(xtrain, ytrain)
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f} seconds")

    # Predict on unseen data
    print("Starting inference…")
    t2 = time.time()
    preds = method_obj.predict(xval)
    t3 = time.time()
    inf_time = t3 - t2
    print(f"Inference time on {len(xval)} samples: {inf_time:.2f} seconds "
          f"({inf_time / len(xval):.4f} s/sample)")

    ## Report results: performance on train and valid/test sets
    preds_train = method_obj.predict(xtrain)
    acc_train = accuracy_fn(preds_train, ytrain)
    macrof1_train = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    preds = method_obj.predict(xval)
    acc_val = accuracy_fn(preds, yval)
    macrof1_val = macrof1_fn(preds, yval)
    print(f"Validation set:  accuracy = {acc_val:.3f}% - F1-score = {macrof1_val:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    if cv_results is not None and cv_param_list is not None:
        plt.figure()
        plt.plot(cv_param_list, cv_results, marker='o')
        if args.cv == 'lr':
            plt.xscale('log')
        if args.cv == 'bs':
            plt.xticks(cv_param_list) 
        plt.title(f'Validation Accuracy vs {args.cv}')
        plt.xlabel(args.cv)
        plt.ylabel('Validation Accuracy')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    # Argument we added 
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for deep networks')
    parser.add_argument('--cv', type=str, choices=['lr', 'dropout', 'bs', 'hidden_dims'], default=None,
                    help="Specify which hyperparameter to cross-validate: 'lr', 'dropout', or 'bs'")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
