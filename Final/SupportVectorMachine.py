import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('survey_lung_cancer.csv')

X = dataset.drop(['AGE','GENDER','SHORTNESS OF BREATH','SMOKING','LUNG_CANCER'],axis=1)
y = dataset['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=42)

SVM = SVC()
SVM.fit(X, y)
predictions = SVM.predict(X_test)
val1 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for SVM: ", val1, "\n")
print("*Confusion Matrix for SVM: ")
print(confusion_matrix(y_test, predictions))
print("*Classification Report for SVM: ")
print(classification_report(y_test, predictions))

y_pred_svm = SVM.predict(X_test)

cm = confusion_matrix(y_test, y_pred_svm)
cm
svm_result = accuracy_score(y_test,y_pred_svm)
print("Accuracy :",svm_result)
recall_svm = cm[0][0]/(cm[0][0] + cm[0][1])
precision_svm = cm[0][0]/(cm[0][0]+cm[1][1])
recall_svm,precision_svm

new_data = pd.DataFrame(pd.read_csv("testdata.csv"))
new_predictions = SVM.predict(new_data)
print(f"prediction : {new_predictions}")


