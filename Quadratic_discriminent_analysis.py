"""# Quadratic Discriminant Analysis #"""

import feature_scaling
X_test = feature_scaling.X_test
y_test = feature_scaling.y_test
X_train = feature_scaling.X_train
y_train = feature_scaling.y_train


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data for demonstration
#X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Quadratic Discriminant Analysis classifier
qda = QuadraticDiscriminantAnalysis()

# Train the classifier on the training set
qda.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = qda.predict(X_test)

# Calculate the accuracy of the classifier
accuracy_qda = accuracy_score(y_test, y_pred)
# print("Accuracy: {:.2f}%".format(accuracy_qda * 100))


'Evaluation'

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict


# calculate and print evaluation metrics
print("Quadratic discriminent analysis")
print("Accuracy: {:.2f}%".format(accuracy_qda * 100))
# print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

# print confusion matrix and classification report
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification report:")
print(classification_report(y_test, y_pred))