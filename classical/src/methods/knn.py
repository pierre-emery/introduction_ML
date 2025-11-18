import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels

        pred_labels = self.predict(self.training_data)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        test_table = []

        for i in range(test_data.shape[0]):
            distances = self.euclidean_dist(test_data[i], self.training_data)
            neighbors_indices = self.find_k_nearest_neighbors(self.k, distances)
            neighbor_labels = self.training_labels[neighbors_indices]

            label = self.predict_label(neighbor_labels)

            test_table.append(label)
        
        test_labels = np.array(test_table)
        return test_labels

    def euclidean_dist(self, example, training_examples):
        """
            Compute the Euclidean distance between a single example
            vector and all training_examples.

            Inputs:
                example: shape (D,)
                training_examples: shape (NxD) 
            Outputs:
                euclidean distances: shape (N,)
        """
        dist = np.sqrt(np.sum((training_examples-example)**2, axis=1))
        return dist
    
    def find_k_nearest_neighbors(self, k, distances):
        """
            Find the indices of the k smallest distances from a list of distances.

            Inputs:
                k: integer
                distances: shape (N,) 
            Outputs:
                indices of the k nearest neighbors: shape (k,)
        """
        indices = np.argsort(distances)[:k]
        return indices
    
    def predict_label(self, neighbor_labels):
        """
            Return the most frequent label in the neighbors'.

        Inputs:
            neighbor_labels: shape (N,) 
        Outputs:
            most frequent label
        """
        neighbor_labels = np.array(neighbor_labels, dtype=int)
    
        count = np.bincount(neighbor_labels)
        most_frequent = np.argmax(count)
        return most_frequent

    