from IPython import get_ipython;get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import funcionesSingleDispatch as fs
from sklearn.metrics import accuracy_score, f1_score, \
    precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import random
from xgboost import XGBClassifier
import funciones as f
import matplotlib.pyplot as plt
from importlib import reload
from sklearn.metrics import plot_roc_curve

reload(f)

PROJPATH = r'C:\Users\jvandam\PycharmProjects\DataScience'

archivo = r'WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(PROJPATH + '\\datasets\\' + archivo, index_col='customerID')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.fillna(df['TotalCharges'].mean(), inplace=True)


# df.isna().sum() -> okis

# df.Churn.value_counts()No     5174; Yes    1869

colsObj = df.select_dtypes(include='object').columns
colsNum = df.select_dtypes(include='number').columns
target = 'Churn'


#
# for i in colsObj:
#     f.categoricoBarras(i, target, df, enPorcentaje='categoria', interactivo=True)
#
# for i in colsNum:
#     f.histogramaPorGruposPandas(i, target, df, interactivo=True)


df = pd.concat([pd.get_dummies(df.loc[:,colsObj], drop_first=True),
                     df.select_dtypes(include='number')], axis=1)

df.columns = [k.replace(" ","_") for k in df.columns]
df.rename(columns={'Churn_Yes':target}, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(df.loc[:,df.columns.drop(target)], df.loc[:,target]
                                                    , test_size=0.25, random_state=80119, stratify=df.loc[:,target])


modeloLogit = LogisticRegression(max_iter=500, random_state=80119, penalty='none')
modeloLogit.fit(X_train,y_train)
y_pred = modeloLogit.predict(X_test)
probs = modeloLogit.predict_proba(X_test)
prob1 = probs[:,1]
print("CLASIF BASE:\n")
print(classification_report(y_test,y_pred))

f.ploteaBarrasModeloBinario(y_test, prob1)

####roc curve

mlPlot = plot_roc_curve(modeloLogit, X_test, y_test)
plt.show()

modeloXGB = XGBClassifier(random_state=80119)
modeloXGB.fit(X_train, y_train)
y_predXGB = modeloXGB.predict(X_test)
print(classification_report(y_test, y_predXGB))


mlXGB = plot_roc_curve(modeloXGB, X_test, y_test)
ax = plt.gca()
plot_roc_curve(modeloXGB, X_test, y_test, ax=ax)
plot_roc_curve(modeloLogit, X_test, y_test, ax=ax)
plt.show()