import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score, cohen_kappa_score, roc_auc_score
import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_excel("Processed_Meningitis_DS.xlsx")

# TDA
Features = ['MCH', 'CBC NEUTROPHILS', 'URINE LEUKOCYTES', 'URINE ERYTHROCYTES']

# Pearson Features
#Features = ['HCT', 'EOSINOPHILS', 'BILIRUBIN_T', 'NA' ,'URINE LEUKOCYTES']

'''
Features = ['WBC', 'RBC', 'Hb%', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLTS', 'CBC NEUTROPHILS', 'LYMPH', 'CBC MONOCYTES',
            'EOSINOPHILS', 'BASOPHILS', 'RDW', 'ESR', 'CRP TITER',
            'BUN', 'CREATININE', 'SGPT', 'SGOT', 'BILIRUBIN_T',
            'BILIRUBIN_D', 'ALK_PHOS', 'RBS', 'CALCIUM', 'MG',
            'ALBUMIN_S', 'NA', 'K', 'CLORIDE', 'S.GRAVITY', 'URINE LEUKOCYTES',
            'URINE ERYTHROCYTES']
'''
feature_cols = ['RBC', 'Hb%', 'HCT', 'MCV', 'MCH', 'MCHC', 'CBC NEUTROPHILS', 'LYMPH', 'RDW',
                'BILIRUBIN_T', 'CALCIUM', 'ALBUMIN_S', 'S.GRAVITY', 'URINE LEUKOCYTES',
                'URINE ERYTHROCYTES']
#Features=['MCH', 'CBC NEUTROPHILS', 'URINE LEUKOCYTES', 'URINE ERYTHROCYTES','HCT', 'EOSINOPHILS', 'BILIRUBIN_T', 'NA']


X = df[Features].values
Y = df['Meningitis']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)  # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_train)
# Model Accuracy, how often is the classifier correct?
print("Train Accuracy:", metrics.accuracy_score(y_train, y_pred))

yhat_classes = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Test Accuracy:", metrics.accuracy_score(y_test, yhat_classes))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=Features, class_names=['NEGATIVE', 'POSITIVE'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Meningitis.png')
Image(graph.create_png())

yhat = y_pred
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)

# kappa
kappa = cohen_kappa_score(y_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_test, yhat_classes)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)

'''
def plt_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
np.set_printoptions(precision=2)
plt.figure()
plt_confusion_matrix(cnf_matrix, classes=['Meningitis=1', 'Not Meninigitis=0'], normalize=False,
                     title='Confusion Matric')
'''