"""# DATA PRE-PROCESSING


"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

# Commented out IPython magic to ensure Python compatibility.
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


df = pd.read_csv(r"D:\CRS_Code\CRS_project\data\Crop_recommendation.csv")
df.head()
print(df)


c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y=df.target
X=df[['N','P','K','temperature','humidity','ph','rainfall']]

"""**Correlation visualization between features. We can see how Phosphorous levels and Potassium levels are highly correlated.**"""

sns.heatmap(X.corr())
plt.show()
