"""
PG19IT02- study.ipynb

# CROP RECOMMENDATION SYSTEM

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

df.describe()

"""# Exploratory Data Analysis

### Heatmap to check null/missing values
"""

sns.heatmap(df.isnull(), cmap="coolwarm")
# plt.show()

"""<h4> <u>Let's have a closer look at the distribution of temperature and ph.</u><br><br>

It is symmetrical and bell shaped, showing that trials will usually give a result near the average, but will occasionally deviate by large amounts. It's also fascinating how these two really resemble each other!</h4>
"""

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.show()

# sns.distplot(df_setosa['sepal_length'],kde=True,color='green',bins=20,hist_kws={'alpha':0.3})
sns.distplot(df['temperature'], color="purple", bins=15, hist_kws={'alpha': 0.2})
plt.subplot(1, 2, 2)
plt.show()

sns.distplot(df['ph'], color="green", bins=15, hist_kws={'alpha': 0.2})
plt.show()

"""<h4> A quick check if the dataset is balanced or not. If found imbalanced, we would have to downsample some targets which are more in quantity but so far everything looks good! <h4>"""

sns.countplot(y='label', data=df, palette="plasma_r")
plt.show()
"""<h4> A very important plot to visualize the diagonal distribution between two features for all the combinations! It is great to visualize how classes differ from each other in a particular space."""

sns.pairplot(df, hue='label')
plt.show()
"""#### During rainy season, average rainfall is high (average 120 mm) and temperature is mildly chill (less than 30'C).

#### Rain affects soil moisture which affects ph of the soil. Here are the crops which are likely to be planted during this season. 

- <b> Rice needs heavy rainfall (>200 mm) and a humidity above 80%. No wonder major rice production in India comes from East Coasts which has average of 220 mm rainfall every year!
- <b> Coconut is a tropical crop and needs high humidity therefore explaining massive exports from coastal areas around the country.
"""

sns.jointplot(x="rainfall", y="humidity", data=df[(df['temperature'] < 30) & (df['rainfall'] > 120)], hue="label")
plt.show()
"""#### This graph correlates with average potassium (K) and average nitrogen (N) value (both>50). 
#### These soil ingredients direcly affects nutrition value of the food. Fruits which have high nutrients typically has consistent potassium values.
"""

sns.jointplot(x="K", y="N", data=df[(df['N'] > 40) & (df['K'] > 40)], hue="label")
plt.show()
"""<h4>Let's try to plot a specfic case of pairplot between `humidity` and `K` (potassium levels in the soil.)</h4>

#### `sns.jointplot()` can be used for bivariate analysis to plot between humidity and K levels based on Label type. It further generates frequency distribution of classes with respect to features
"""

sns.jointplot(x="K", y="humidity", data=df, hue='label', size=8, s=30, alpha=0.7)
plt.show()
"""#### We can see ph values are critical when it comes to soil. A stability between 6 and 7 is preffered"""

sns.boxplot(y='label', x='ph', data=df)
plt.show()
"""#### Another interesting analysis where Phosphorous levels are quite differentiable when it rains heavily (above 150 mm)."""

sns.boxplot(y='label', x='P', data=df[df['rainfall'] > 150])
plt.show()
"""#### Further analyzing phosphorous levels.

When humidity is less than 65, almost same phosphor levels(approx 14 to 25) are required for 6 crops which could be grown just based on the amount of rain expected over the next few weeks.
"""

sns.lineplot(data=df[(df['humidity'] < 65)], x="K", y="rainfall", hue="label")
plt.show()