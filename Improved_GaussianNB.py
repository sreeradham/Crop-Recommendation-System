from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
import pandas as pd

# Load iris dataset
# iris = load_iris()
# X, y = iris.data, iris.target

# df = pd.read_csv(r'D:\SREE\Project\yield_prediction\data\Crop_recommendation.csv')
df = pd.read_csv(r"D:\CRS_Code\CRS_project\data\Crop_recommendation1.csv")
# Split dataset into features and target
X = df.drop("label", axis=1)
y = df["label"]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base estimator
base_estimator = GaussianNB()

# Define the bagging classifier
bagging_classifier = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

# Fit the bagging classifier to the training set
bagging_classifier.fit(X_train, y_train)

# Evaluate the bagging classifier on the test set
score = bagging_classifier.score(X_test, y_test)

print("Test set score: ", score*100)