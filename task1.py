import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = np.loadtxt('ex1data1.txt', delimiter=',')

# Step 2: Visualize the Data
population = data[:, 0]
profit = data[:, 1]
plt.scatter(population, profit, marker='.', color='blue')
plt.xlabel('Population of City (in 10,000s)')
plt.ylabel('Profit (in $10,000s)')
plt.title('Food Truck Profits vs. City Population')
plt.show()

# Step 3: Prepare the Data
X = np.column_stack((np.ones_like(population), population))  # Add intercept term
y = profit

# Step 4: Define the Model
def linear_regression(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

# Step 5: Train the Model
theta = linear_regression(X, y)

# Step 6: Make Predictions
new_population = 35  # Example: New city population
predicted_profit = np.array([1, new_population]) @ theta
print(f'For a city with a population of {new_population} (in 10,000s), the predicted profit is ${predicted_profit * 10000:.2f}')

# Step 7: Evaluate the Model (Optional)
# Calculate mean squared error, visualize regression line, etc.
