import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

df = pd.read_csv(r"D:\CRS_Code\CRS_project\data\Crop_recommendation.csv")

c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y = df.target
X = df[['N','P','K','temperature','humidity','ph','rainfall']]

"""# FEATURE SCALING
Two of our features (temperature and ph) are gaussian distributed, therefore scaling them between 0 and 1 with MinMaxScaler.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)

# print(X_train, X_test, y_train, y_test)