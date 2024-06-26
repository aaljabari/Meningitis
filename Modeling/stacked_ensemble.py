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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE  # Import SMOTE

# ====================== Prepair Dataset =============================
df = pd.read_excel("Modeling/Data/0-Final_Minigitis_Dataset.xlsx")
Gini_features=["WBC (2004)","RBC (2004)","HCT (2004)","MCHC (2004)","PLTS (2004)","EOSINOPHILS (2004)"]
Features = ['MCH', 'CBC MONOCYTES', 'ESR', 'BILIRUBIN_D', 'ALK_PHOS']
SAFE_Features = ['ERYTHROCYTES (6001)', 'LEUKOCYTES (6001)', 'NEUTROPHILS (2004)', 'MCH (2004)']
Pearson_Feature = ['URINE ERYTHROCYTES', 'NA', 'BILIRUBIN_T', 'EOSINOPHILS', 'HCT']
Mixed_Features = ['URINE ERYTHROCYTES', 'URINE LEUKOCYTES', 'CBC NEUTROPHILS', 'MCH', 'NA', 'BILIRUBIN_T',
                  'EOSINOPHILS', 'HCT']
X = df[SAFE_Features].values
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
# ====================================================================

base_learners = [
    ('l1', KNeighborsClassifier(n_neighbors=10)),
    ('l2', DecisionTreeClassifier(min_samples_split=70, max_depth=15)),
    ('l3', SVC(gamma=2, C=1)),
    ('l4', RandomForestClassifier(min_samples_split=90, max_depth=9)),
    #('l5', MLPClassifier(max_iter=100, learning_rate='constant', hidden_layer_sizes=(20, 7, 3), activation='tanh'))
]
model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(), cv=5)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Training Accuracy ", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy", accuracy_score(y_test, y_test_pred))

print(classification_report(y_test, y_test_pred))
cf_matrix = confusion_matrix(y_test, y_test_pred)

plt.subplots(figsize=(8, 5))

sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")

plt.show()
