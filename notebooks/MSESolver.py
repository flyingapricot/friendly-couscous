import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def predict_and_calculate_mse(X, y, new_data_point):
    """
    Train a linear regression model, predict the output for a new data point,
    and calculate the Mean Squared Error (MSE) on the training data.

    Parameters:
    X (array-like): Feature matrix of shape (n_samples, n_features).
    y (array-like): Target matrix of shape (n_samples,).
    new_data_point (array-like): New data point of shape (1, n_features).

    Returns:
    float: Predicted output for the new data point.
    float: Mean Squared Error (MSE) of the model on the training data.
    """
    # Ensure inputs are numpy arrays
    X = np.array(X)
    y = np.array(y)
    new_data_point = np.array(new_data_point)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the output for the new data point
    predicted_output = model.predict(new_data_point)

    # Calculate the Mean Squared Error (MSE) on the training data
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    return predicted_output[0], mse
