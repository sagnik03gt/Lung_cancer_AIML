import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('survey_lung_cancer.csv')

X = dataset.drop(['AGE','GENDER','SHORTNESS OF BREATH','SMOKING','LUNG_CANCER'],axis=1)
y = dataset['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=42)

KNN = KNeighborsClassifier()
KNN.fit(X, y)
predictions = KNN.predict(X_test)
val3 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for KNN: ", val3, "\n")
print("*Confusion Matrix for KNN: ")
print(confusion_matrix(y_test, predictions))
print("*Classification Report for KNN: ")
print(classification_report(y_test, predictions))

y_pred_knn = KNN.predict(X_test)
cm = confusion_matrix(y_test, y_pred_knn)
cm
knn_result = accuracy_score(y_test,y_pred_knn)
print("Accuracy :",knn_result)
recall_knn = cm[0][0]/(cm[0][0] + cm[0][1])
precision_knn = cm[0][0]/(cm[0][0]+cm[1][1])
recall_knn,precision_knn

new_data = pd.DataFrame(pd.read_csv("testdata.csv"))
new_predictions = KNN.predict(new_data)
print(f"prediction : {new_predictions}")