import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = np.loadtxt('ex1data2.txt', delimiter=',')

# Step 2: Optionally visualize the data
# Size of the house (in square feet)
house_size = data[:, 0]
# Number of bedrooms
num_bedrooms = data[:, 1]
# House price
house_price = data[:, 2]

# Step 3: Prepare the data
X = np.column_stack((np.ones_like(house_size), house_size, num_bedrooms))  # Add intercept term
y = house_price

# Step 4: Optionally, feature scaling
# Uncomment the following lines to scale the features
# X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)

# Step 5: Define the model
def linear_regression(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

# Step 6: Train the model
theta = linear_regression(X, y)

# Step 7: Make predictions
new_house_size = 1650  # Example: New house size
new_num_bedrooms = 3  # Example: Number of bedrooms for the new house
new_house_features = np.array([1, new_house_size, new_num_bedrooms])
predicted_price = new_house_features @ theta
print(f'The predicted price for a house with {new_house_size} square feet and {new_num_bedrooms} bedrooms is ${predicted_price:.2f}')

# Step 8: Evaluate the model (Optional)
# Calculate mean squared error, visualize regression line, etc.
