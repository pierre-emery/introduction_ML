import numpy as np

from ..utils import  accuracy_fn, label_to_onehot



class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        training_labels_onehot = label_to_onehot(training_labels)
    
        self.weights = np.random.normal(0., 0.1, [training_data.shape[1], training_labels_onehot.shape[1]])

        for it in range(self.max_iters):        # Compute the gradient and perform a gradient step

            gradient = self.gradient_logistic_multi(training_data, training_labels, self.weights)
            self.weights -= self.lr * gradient

            predictions = self.predict(training_data)  
            if accuracy_fn(predictions, np.argmax(training_labels_onehot, axis=1)) == 100:  # If accuracy hits 100%, stop early.
                break

            if it % 100 == 0:  # Log the loss every 100 iterations
                print(f"Loss at iteration {it}: {self.loss_logistic_multi(training_data, training_labels_onehot, self.weights)}")

       
        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
        Prediction the label of data for multi-class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        logits = np.dot(test_data, self.weights)
        pred_labels = np.argmax(logits, axis=1)  
        return pred_labels
    
    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic regression.

        Arguments:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the range [0, 1]
        """
        logits = np.dot(data, W)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return softmax_probs
    
    def loss_logistic_multi(self, data, labels, w):
        """ 
        Loss function for multi-class logistic regression, i.e., multi-class entropy.
        
        Arguments:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value 
        """
        logits = np.dot(data, w)
        softmax_probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_probs /= np.sum(softmax_probs, axis=1, keepdims=True)
        
        loss = -np.sum(labels * np.log(softmax_probs + 1e-10))  
        return loss
    
    def gradient_logistic_multi(self, data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        labels_onehot = label_to_onehot(labels, W.shape[1])

        logits = np.dot(data, W)
        softmax_probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_probs /= np.sum(softmax_probs, axis=1, keepdims=True)

        grad = np.dot(data.T, (softmax_probs - labels_onehot)) 
        return grad
    
    
    
    
    
