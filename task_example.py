import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Function to reduce dimensionality using random projection
def reduce_dimensionality(X, target_dimension):
    random_projector = SparseRandomProjection(n_components=target_dimension, random_state=42)
    X_transformed = random_projector.fit_transform(X)
    return X_transformed

# Function to visualize the dataset before and after dimensionality reduction
def visualize_dataset(X_original, X_transformed, target_dimension):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_original[:, 0], X_original[:, 1], c=X_original[:, 2], cmap='viridis', marker='o', edgecolors='k')
    plt.title('Original Dataset')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.colorbar(label='Petal Length')
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=X_original[:, 2], cmap='viridis', marker='o', edgecolors='k')
    plt.title(f'Reduced Dimensionality (Target Dimension: {target_dimension})')
    plt.xlabel('Random Projection Component 1')
    plt.ylabel('Random Projection Component 2')
    plt.colorbar(label='Petal Length')
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=X_original[:, 3], cmap='viridis', marker='o', edgecolors='k')
    plt.title(f'Reduced Dimensionality (Target Dimension: {target_dimension})')
    plt.xlabel('Random Projection Component 1')
    plt.ylabel('Random Projection Component 2')
    plt.colorbar(label='Petal Width')
    
    plt.tight_layout()
    plt.show()

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the original dataset
visualize_dataset(X_train, X_train, target_dimension=X_train.shape[1])

# Set the target dimension for reduction
target_dimension = 2

# Reduce the dimensionality of the training and testing sets
X_train_reduced = reduce_dimensionality(X_train, target_dimension)
X_test_reduced = reduce_dimensionality(X_test, target_dimension)

# Visualize the dataset after dimensionality reduction
visualize_dataset(X_train, X_train_reduced, target_dimension=target_dimension)

# Train a simple classifier on the reduced dataset
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train_reduced, y_train)

# Predict on the reduced test set
y_pred = classifier.predict(X_test_reduced)

# Calculate accuracy on the reduced test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the reduced test set: {accuracy:.2f}')