import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


PROJPATH = r'C:\Users\jvandam\PycharmProjects\DataScience'

df = pd.read_csv(PROJPATH + '/datasets/breast-cancer.data', header=None)

df.columns = ['target' , 'age' , 'menopause' , 'tumor' , 'inv' , 'node' , 'deg' , 'breast' , 'breast_quad' , 'irradiat' ]

X = df.iloc[:,1:10]
X = X.astype(str)
Y = df.iloc[:,0]
# encode string input values as integers
columns = []
for i in range(0, X.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(X.iloc[:,i])
    feature = feature.reshape(X.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    columns.append(feature)


encoded_x = np.column_stack(columns)

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y,test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)

predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

###cross val score###

score_cv = cross_validate(model, X_train, y_train, cv=5, scoring=['precision_macro', 'recall_micro'])

score_cv['test_precision_macro']
