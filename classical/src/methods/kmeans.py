import numpy as np

from ..utils import  accuracy_fn

class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, K=3, max_iters=1000):
        """
        Call set_arguments function of this class.
        """
        self.K = K
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, data, labels=None, n_init=10):
        """
        Trains the KMeans algorithm with multiple random initializations and returns predicted labels for training data.
        
        Arguments:
            data (np.array): Input data of shape (N, D) where N is the number of samples and D is the number of features.
            labels (np.array, optional): Labels of shape (N,) (unused for KMeans, but added to support existing call).
            n_init (int): The number of random initializations to try.
            
        Returns:
            pred_labels (np.array): The predicted cluster labels for each point in the training data.
            best_centroids (np.array): The final best cluster centers after multiple initializations.
        """
        best_centroids = None
        best_cluster_assignments = None
        best_ssd = float('inf')  # Set initial best SSD to infinity

        for init in range(n_init):

            # Initialize the centers for this run
            self.centroids = self.init_centers(data)

            # Loop over the iterations (as in k_means)
            for i in range(self.max_iters):

                old_centroids = self.centroids.copy()  # Keep a copy of the centroids from the previous iteration
                
                # Compute the distances between data points and centroids
                distances = self.compute_distance(data, self.centroids)
                cluster_assignments = self.find_closest_cluster(distances)  # Find the closest cluster for each point

                # Update centroids
                self.centroids = self.compute_centers(data, cluster_assignments, self.K)

                # End the algorithm if the centroids have not changed (convergence check)
                if np.allclose(self.centroids, old_centroids):
                    break
            
            # Calculate SSD for this initialization
           
            ssd = self.sum_squared_differences(data, self.centroids, cluster_assignments)  # SSD is the sum of the minimal distances to the closest cluster center
            

            # Track the best initialization (lowest SSD)
            if ssd < best_ssd:
                best_ssd = ssd
                best_centroids = self.centroids
                best_cluster_assignments = cluster_assignments

        self.centroids = best_centroids
        cluster_center_label = self.assign_labels_to_centers(self.centroids, best_cluster_assignments, labels)

        pred_labels = self.predict_with_centers(data, self.centroids, cluster_center_label)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        distances = self.compute_distance(test_data, self.centroids)
        test_labels = self.find_closest_cluster(distances)

        return test_labels
    
    def init_centers(self, data):
        """
        Randomly pick K data points from the data as initial cluster centers.
        
        Arguments: 
            data: array of shape (N, D) where N is the number of data points and D is the number of features.
        Returns:
            centers: array of shape (K, D), the initial cluster centers.
        """
        random_idx = np.random.permutation(data.shape[0])[:self.K]  # Randomly permute the indices and select the first K
        centers = data[random_idx]  # Select the K data points as initial centroids
        
        return centers
    
    
    def compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.
        
        Arguments:    
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """
        N = data.shape[0]
        K = centers.shape[0]
        distances = np.zeros((N, K))
        for k in range(K):
            # Compute the Euclidean or manhattan distance for each data to each center
            center = centers[k]
            distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1)) #Euclidian
            """distances[:, k] = np.abs(data - center).sum(axis=1)""" # Manhattan
            
        return distances
    
    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.
        
        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments
    
    def compute_centers(self, data, cluster_assignments, K):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments: 
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        centers = []
        for k in range(K):
            cluster_points = data[cluster_assignments == k]
            
            if cluster_points.shape[0] > 0:  # If the cluster has points assigned
                centers.append(cluster_points.mean(axis=0))
            else:  
                centers.append(centers[k])  # If the cluster is empty, keep the previous center
            
        return np.array(centers)
    
    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments: 
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns: 
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        cluster_center_label = np.zeros(centers.shape[0], dtype=int)  # Initialize an array to store the cluster labels
        for i in range(len(centers)):
            cluster_labels = true_labels[cluster_assignments == i]
            cluster_labels = cluster_labels.astype(int) 
            if len(cluster_labels) > 0:
                label = np.argmax(np.bincount(cluster_labels))
            else:
                label = -1  
            cluster_center_label[i] = label
        return cluster_center_label
    
        

    def predict_with_centers(self, data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assign points in data to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments: 
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns: 
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """
        distances = self.compute_distance(data, centers)
        cluster_assignments = self.find_closest_cluster(distances)

        new_labels = cluster_center_label[cluster_assignments]
        return new_labels
    
    def sum_squared_differences(self, X, centroids, cluster_assignments):
        """
        Calculate the Sum of Squared Differences (SSD) between the data points and their assigned centroids.
        
        Arguments:
            X (np.array): The data points of shape (N, D)
            centroids (np.array): The centroids of the clusters of shape (K, D)
            cluster_assignments (np.array): The cluster assignments of shape (N,)

        Returns:
            ssd (float): The sum of squared differences between points and their centroids
        """
        ssd = 0
        K = centroids.shape[0]
        
        for i in range(K):
            # Find the points assigned to cluster i
            cluster_points = X[cluster_assignments == i]
            
            # Compute the squared distances from points to the centroid
            ssd += np.sum(np.square(cluster_points - centroids[i]))
            
        return ssd
    
    def average_wcss(self, X, labels, max_k=10, n_init=10):
        """
        Compute the average WCSS for multiple initializations and different values of k (number of clusters).
        
        Arguments:
            X (np.array): The data to be clustered (shape: NxD).
            labels (np.array): The true labels for the data (shape: Nx1).
            max_k (int): The maximum number of clusters to consider for k-means.
            n_init (int): The number of initializations to perform for each k.
        
        Returns:
            wcss (list): List of average WCSS values for each k.
        """
        wcss = []  # List to store WCSS for each k

        for k in range(1, max_k + 1):
            self.K = k  # Update k for each iteration
            
            all_wcss = []  # List to store WCSS for multiple initializations of k-means
            for _ in range(n_init):
                # Pass labels to fit
                pred_labels = self.fit(X, labels)  # Fit the model and get predicted labels
                
                # Compute WCSS (SSD) for this initialization
                wcss_value = self.sum_squared_differences(X, self.centroids, pred_labels)
                all_wcss.append(wcss_value)
            
            # Calculate the average WCSS for this value of k
            avg_wcss = np.mean(all_wcss)/k
            wcss.append(avg_wcss)
        
        return wcss