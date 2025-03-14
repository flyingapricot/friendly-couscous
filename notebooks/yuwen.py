import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split
from numpy.linalg import inv, det, matrix_rank
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

def check_invertibility(X):
    '''
    X is a (m,d) matrix. m is samples, d is features
    outputs true false values for (left_inverse_exist,right_inverse_exists)
    doesnt check exactly equal to 0 as det might be vv small
    '''
    use_LI = True if (det(X.T @X) - 0.001 > 0) else False
    use_RI = True if (det(X @ X.T) - 0.001 > 0) else False
    return (use_LI,use_RI)

def apply_left_inverse(X, y ,L = 0):
    '''
    X as a (m,d) matrix must be overdetermined: m > d
    y is a d-tuple
    assumes that X^T @ X is invertible
    L is the value of Lambda (regularization hyperparameter)
    '''
    (m,d) = X.shape
    w = inv(X.T @ X + L * np.eye(d)) @ X.T @ y
    return w

def apply_right_inverse(X,y,L = 0):
    '''
    X is a 2d (m,d)-ndarray
    X as a (m,d) matrix must be under-determined: m < d
    y is a 1d (d)-ndarray
    assumes that X @ X^T  is invertible
    L is the value of Lambda (regularization hyperparameter)
    '''
    (m,d) = X.shape
    w = X.T @ inv(X @ X.T + L * np.eye(m)) @ y
    return w

def aug_first_coln(X):
    # creates a vertical vector of 1s, size is the no of rows in X
    ones = np.ones((X.shape[0], 1))
    # places the vertical row of ones in the first coln of X
    X_aug = np.hstack((ones, X))
    return X_aug

def get_mse(pred, actual):
    # calculates the MSE between predicted values and actual (expected values)
    MSE = np.square(pred - actual).mean()
    return MSE

def onehot_encode(y):
    '''
    only to convert from classes to onehot
    eg. y = np.array([1,1,2,3,2]).reshape(-1,1)
    '''
    # converts the target into onehot encoding format
    onehot_encoder=OneHotEncoder(sparse_output=False)

    # converts from classes to one hot encoding 
    Ytr_onehot = onehot_encoder.fit_transform(y)
    return Ytr_onehot

def output_to_one_hot(predictions):
    """
    Converts the output predictions into one-hot encoding format.
    If 1d array aka only one sample, reshape it into 1 row
    Args: predictions (list or np.ndarray): A 2D list or array where each row contains predicted scores or probabilities for each class.
    Returns: np.ndarray: A 2D array in one-hot encoding format.
    """
    predictions = np.array(predictions)  # Ensure input is an array
    # Find the index of the max value in each row
    max_indices = np.argmax(predictions, axis=1)
    # Create a one-hot encoded array
    one_hot = np.zeros_like(predictions)
    one_hot[np.arange(predictions.shape[0]), max_indices] = 1
    return one_hot

def split_dataset(data, target, test_size):
    # splits the dataset into training data and test data
    # 0 <= test_size <= 1 to indicate proportion of data to be allocated as test data
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size)
    return (X_train, X_test, y_train, y_test)

def calc_expectation(outcomes, probability):
    # takes in 2 np arrays and find the expectation
    outcomes = np.asarray(outcomes)
    probability = np.asarray(probability)
    return (outcomes * probability).sum() / probability.sum()

def calc_variance(outcomes, probabilities):
    """Calculate the variance of a discrete random variable."""
    mean = calc_expectation(outcomes, probabilities)
    variance = sum(prob * (outcome - mean) ** 2 for outcome, prob in zip(outcomes, probabilities))
    return variance

def calc_standard_deviation(outcomes, probabilities):
    """Calculate the standard deviation of a discrete random variable."""
    variance = calc_variance(outcomes, probabilities)
    return np.sqrt(variance)

def matrix_cofactor(matrix):
 
    try:
        determinant = np.linalg.det(matrix)
        if(determinant!=0):
            cofactor = None
            cofactor = np.linalg.inv(matrix).T * determinant
            # return cofactor matrix of the given matrix
            return cofactor
        else:
            raise Exception("singular matrix")
    except Exception as e:
        print("could not find cofactor matrix due to",e)

def sigmoid(a):
    """ this function implements the sigmoid function, and 
    expects a numpy array as argument """
    
    if not isinstance(a, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    sigmoid = 1.0/(1.0 + np.exp(-a))
    return sigmoid 

def find_gini(prob):
    '''
    takes in a prob list, all elements must sum to 1
    '''
    return 1 - sum(x*x for x in prob)

def find_entropy(prob):
    '''
    takes in probability list for each class
    math.log(value, base)
    '''
    return sum(-1 * x * math.log(x, 2) for x in prob)

def find_misclassification(prob):
    '''
    takes in probability list for each class
    '''
    return 1 - max(prob)

def find_weighted_average(weight, prob):
    '''
    takes in weights and probability list for each class
    '''
    if len(weight) != len(prob):
        raise ValueError("Weight and probability lists must have the same length")
    return sum(x * y for x,y in zip(weight, prob))
