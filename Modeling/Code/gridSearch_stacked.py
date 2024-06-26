import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ====================== Prepare Dataset =============================
df = pd.read_excel("Modeling/Data/0-Final_Minigitis_Dataset.xlsx")
SAFE_Features = ['ERYTHROCYTES (6001)', 'LEUKOCYTES (6001)', 'NEUTROPHILS (2004)', 'MCH (2004)']
X = df[SAFE_Features].values
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
# ====================================================================

# Define the base learners
base_learners = [
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC()),
    ('rf', RandomForestClassifier())
]

# Define the Stacking Classifier
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5
)

# Define parameter grids for each base learner and the final estimator
param_grid = {
    'knn__n_neighbors': [5, 10, 15],
    'dt__min_samples_split': [2, 50, 100],
    'dt__max_depth': [None, 10, 20],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto'],
    'rf__n_estimators': [50, 100, 200],
    'rf__min_samples_split': [2, 50, 100],
    'rf__max_depth': [None, 10, 20],
    'final_estimator__C': [0.1, 1, 10]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=stacking_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Retrieve the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Print results
print("Best parameters found: ", grid_search.best_params_)
print("Training Accuracy: ", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy: ", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Plot confusion matrix
cf_matrix = confusion_matrix(y_test, y_test_pred)
plt.subplots(figsize=(8, 5))
sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")
plt.show()
