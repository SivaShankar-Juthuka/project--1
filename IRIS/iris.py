from sklearn.datasets import load_iris

# Loading the IRIS dataset from sklearn module
iris = load_iris()

# seggregating the independent variables and Dependent variables
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

# Spliting the dataset into training and testing sets in 80 : 20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

from sklearn.neighbors import KNeighborsClassifier

# Create and train the KNN classifiervirginica
k = 3  # Number of neighbors to consider #Setosa, Virtusa, Virginica
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)
