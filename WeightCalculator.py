import numpy as np

# 1.ORC 2.OPTIMAR 3.MAK 
election1_actual = np.array([51.41, 48.59])
election2_actual = np.array([40.87, 24.95, 16.29, 13.12])
election3_actual = np.array([52.59, 30.64, 8.40, 7.29, 0.89, 0.20])

#Polls for each company.
election1_polls = np.array([
    [62., 38.],
    [53.6, 46.4],
    [59.8, 40.2]
])

election2_polls = np.array([
    [50., 23.2, 13.5, 8.4],
    [50.6, 24.7, 14.5, 6.8],
    [50.2, 26.1, 12.6, 6.1]
])

election3_polls = np.array([
    [53.4, 23.8, 11.5, 8.7, 2.1, 0.5],
    [51.6, 28., 8.5, 10.3, 1.1, 0.5],
    [51.5, 24.4, 12.5, 8.8, 2.2, 0.6]
])

# Initial weights.
weights = np.array([0.3, 0.3, 0.3])

# Learning rate and iterations. The results get more accurate with more iterations.
learning_rate = 0.0001
num_iterations = 150000

def mean_squared_error(actual, predicted):
    return np.mean((actual - predicted) ** 2)

for i in range(num_iterations):
    # Make predictions using the current weights for each election
    election1_predictions = np.dot(election1_polls.T, weights)
    election2_predictions = np.dot(election2_polls.T, weights)
    election3_predictions = np.dot(election3_polls.T, weights)

    error1 = mean_squared_error(election1_actual, election1_predictions)
    error2 = mean_squared_error(election2_actual, election2_predictions)
    error3 = mean_squared_error(election3_actual, election3_predictions)
    error = (error1 + error2 + error3) / 3  # average error across elections
    
    election1_gradient = np.dot(election1_polls, election1_predictions - election1_actual) / len(election1_actual)
    election2_gradient = np.dot(election2_polls, election2_predictions - election2_actual) / len(election2_actual)
    election3_gradient = np.dot(election3_polls, election3_predictions - election3_actual) / len(election3_actual)
    
    weights -= learning_rate * (election1_gradient + election2_gradient + election3_gradient) / 3


    if i % 5000 == 0:
        print(f"Iteration {i}: MSE = {error:.6f}", "Weights:", weights)

print("Final weights:", weights)