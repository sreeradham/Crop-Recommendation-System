import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"D:\CRS_Code\CRS_project\data\Crop_recommendation.csv")


c = df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target'] = c.cat.codes

y = df.target
X = df[['N','P','K','temperature','humidity','ph','rainfall']]

print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)