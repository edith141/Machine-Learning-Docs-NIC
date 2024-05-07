from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
print(housing.frame.head())


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate mean squared error (MSE) on the testing data
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



# Visualisation 
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted', alpha=0.2)  # Predicted values in blue
plt.scatter(y_test, y_test, color='red', label='True', alpha=0.2)  # True values in red
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values (Linear Regression)')
plt.legend()
plt.grid(True)
plt.show()
