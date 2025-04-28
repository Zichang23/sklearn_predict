# laod libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn import metrics

# load the dataset
crops = pd.read_csv("soil_measures.csv")

# overview of the data
crops.head()
crops.describe()

# view counts of each crop category
crops.value_counts('crop')

# getting unique values of the crop column
crop_name = crops['crop'].unique()
print(crop_name)

# Data Exploration
# draw barplot
fig, ax = plt.subplots()

for i in crop1:
    crop_df = crops[crops['crop']==i]
    ax.bar(i, crop_df['N'].mean())
ax.set_ylabel("Nitrogen content ratio")
ax.set_xticklabels(crop1, rotation=90)
plt.show()

# create boxplot
ax = sns.boxplot(x='crop', y='ph', data=crops)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('Boxplots by Crops')

# add axis labels
ax.set_xlabel('Crops')
ax.set_ylabel('pH (values of the soil)')

# show the plot
plt.show()

# split data
X= crops.iloc[:,0:3]
y = crops.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify = y)

# fit KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))
train_accuracies = {}
test_accuracies = {}

# visualize training and testing accuracies
neighbors = np.arange(1,26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

plt.figure(figsize=(8,6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()

# evaluate model performance
y_knn_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_knn_pred))
print(classification_report(y_test, y_knn_pred))

# cross-validation
parameters = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print(grid.best_score_)
print(grid.best_params_)

# fit decision tree model
dt = DecisionTreeClassifier(random_state = 42)
dt_classifier.fit(X_train, y_train)
y_dt_pred = dt_classifier.predict(X_test)

# fit logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_logreg_pred = logreg.predict(X_test)

# predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]
print(y_pred_probs[0])

# scale and evaluate classification models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(),
"Decision Tree": DecisionTreeClassifier()}
results = []
for model in models.values():
    kf = KFold(n_splits = 50, random_state = 50, shuffle = True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()

# check test set performance
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print("{} Test Set Accuracy: {}".format(name, test_score))
