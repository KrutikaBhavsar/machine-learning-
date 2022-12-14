import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    mean = np.sum(y, axis=0) / len(y)
    return -np.sum(mean * np.log(mean + EPS))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    mean = np.sum(y, axis=0) / len(y)
    return 1 - np.sum(np.square(mean))
    
def variance(y):
    """
    Computes the variance of the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    return np.var(y)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, mean=0):
        self.feature_index = feature_index
        self.value = threshold
        self.mean = mean
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        self.criterion = self.all_criterions[criterion_name][0]

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j >= threshold
        """

        X_left = X_subset[np.nonzero(X_subset[:, feature_index] < threshold)]
        X_right = X_subset[np.nonzero(X_subset[:, feature_index] >= threshold)]
        y_left = y_subset[np.nonzero(X_subset[:, feature_index] < threshold)]
        y_right = y_subset[np.nonzero(X_subset[:, feature_index] >= threshold)]        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """
        y_left = y_subset[np.nonzero(X_subset[:, feature_index] < threshold)]
        y_right = y_subset[np.nonzero(X_subset[:, feature_index] >= threshold)]
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        col = []
        for i in range(X_subset.shape[1]):
            u_col = np.unique(X_subset[:, i])
            b_col, best_threshold = self.criterion(y_subset), 0
            for j in range(len(u_col) - 1):
                threshold = (u_col[j] + u_col[j + 1]) / 2
                y_left, y_right = self.make_split_only_y(i, threshold, X_subset, y_subset)
                curr_col = (len(y_left) * self.criterion(y_left) + len(y_right) * self.criterion(y_right)) / len(y_subset)
                if curr_col < b_col:
                    b_col, best_threshold = curr_col, threshold
            col.append((b_col, best_threshold))
        col = np.asarray(col)
        feature_index = np.argmin(col, axis=0)[0]
        return feature_index, col[feature_index, 1]
    
    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        y_sub = np.zeros(shape=y_subset.shape)
        y_sub[:, 0] = 1 

        if depth >= self.max_depth or len(y_subset) <= self.min_samples_split or self.criterion(y_subset) < self.criterion(y_sub):
            pred =  0
            if self.classification:
                pred = np.argmax(np.sum(y_subset, axis=0))
                mean = np.sum(y_subset, axis=0) / len(y_subset)
                return Node(0, pred, mean)
            else:
                if self.criterion_name == 'variance':
                    pred = np.mean(y_subset)
                else:
                    pred = np.median(y_subset)
                return Node(0, pred)
        
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)
        left_subsets, right_subsets = self.make_split(feature_index, threshold, X_subset, y_subset)
        new_node.left_child = self.make_tree(*left_subsets, depth + 1)
        new_node.right_child = self.make_tree(*right_subsets, depth + 1)
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the preds should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        X_row = X.shape[0]
        y_predicted = np.ndarray(X_row)
        for i in range(X_row):
            node = self.root
            while node.left_child is not None or node.right_child is not None:
                if X[i][node.feature_index] < node.value and node.left_child is not None:
                    node = node.left_child
                elif node.right_child is not None:
                    node = node.right_child
                y_predicted[i] =  node.value
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class meanbilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the preds should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            meanbilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'
        
        X_row = X.shape[0]
        y_predicted_probs = np.ndarray((X_row, self.n_classes))
        cnt = 0
        for row in X:
            node = self.root
            while node.left_child is not None or node.right_child is not None:
                if row[node.feature_index] < node.value and node.left_child is not None:
                    node = node.left_child
                elif node.right_child is not None:
                    node = node.right_child
            y_predicted_probs[cnt] = node.mean
            cnt = cnt + 1

        return y_predicted_probs
