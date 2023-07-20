from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

np.random.seed(0)

# Creating an object called iris with the iris data
iris = load_iris()

# Creating a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Adding a new column for the species name
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Creating Test and Train Data
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75

train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# Create a list of the feature column's names
features = df.columns[:4]

# Converting each species name into digits 
y = pd.factorize(train['species'])[0]
# print(y)

# Creating a random forest classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Training the classifier
clf.fit(train[features], y)

# Applying the trained Classifier to the test
clf.predict(test[features])

# Viewing the predicted probabilities of the first 10 observations
# print(clf.predict_proba(test[features]))[10:20]

# mapping names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]

# View the PREDICTED species for the first five observations
print(preds[:5])

# Viewing the ACTUAL species for the first five observations
print(test['species'].head())

preds = iris.target_names[clf.predict([[5.0, 3.6, 1.4, 2.0], [5.0, 3.6, 1.4, 2.0]])]
print(preds)