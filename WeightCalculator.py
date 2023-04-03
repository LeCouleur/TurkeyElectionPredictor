import numpy as np

# 1.ORC 2.OPTIMAR 3.MAK 
election1_actual = np.array([51.41, 48.59])
election2_actual = np.array([40.87, 24.95, 16.29, 13.12])
election3_actual = np.array([52.59, 30.64, 8.40, 7.29, 0.89, 0.20])

# Define the poll company predictions for each election
election1_polls = np.array([
    [0.62, 0.38],
    [0.536, 0.464],
    [0.598, 0.402]
]) * 100
election2_polls = np.array([
    [0.50, 0.232, 0.135, 0.084],
    [0.506, 0.247, 0.145, 0.068],
    [0.502, 0.261, 0.126, 0.061]
]) * 100
election3_polls = np.array([
    [0.534, 0.238, 0.115, 0.087, 0.021, 0.005],
    [0.516, 0.28, 0.085, 0.103, 0.011, 0.005],
    [0.515, 0.244, 0.125, 0.088, 0.022, 0.006]
]) * 100

# Define initial weights for each poll company
weights = np.array([0.3, 0.3, 0.3])

# Define the learning rate and number of iterations
learning_rate = 0.0001
num_iterations = 200000

# Define a function to calculate the mean squared error
def mean_squared_error(actual, predicted):
    return np.mean((actual - predicted) ** 2)

# Use gradient descent to update the weights for each election separately
for i in range(num_iterations):
    # Make predictions using the current weights for each election
    election1_predictions = np.dot(election1_polls.T, weights)
    election2_predictions = np.dot(election2_polls.T, weights)
    election3_predictions = np.dot(election3_polls.T, weights)

    # Calculate the mean squared error for each election separately
    error1 = mean_squared_error(election1_actual, election1_predictions)
    error2 = mean_squared_error(election2_actual, election2_predictions)
    error3 = mean_squared_error(election3_actual, election3_predictions)
    error = (error1 + error2 + error3) / 3  # average error across elections

    # Calculate the gradient for each election separately
    election1_gradient = np.dot(election1_polls, election1_predictions - election1_actual) / len(election1_actual)
    election2_gradient = np.dot(election2_polls, election2_predictions - election2_actual) / len(election2_actual)
    election3_gradient = np.dot(election3_polls, election3_predictions - election3_actual) / len(election3_actual)

    # Update the weights for each election separately
    weights -= learning_rate * (election1_gradient + election2_gradient + election3_gradient) / 3

    # Print the error every 5000 iterations
    if i % 5000 == 0:
        print(f"Iteration {i}: MSE = {error:.6f}")

print("Final weights:", weights)
