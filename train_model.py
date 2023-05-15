import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('dataset.csv')

X_train, X_test, y_train, y_test = train_test_split(data[['level','age','gender','income','debt_ratio']], data[['reason', 'method']], test_size=0.2)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

joblib.dump(rfc, 'random_forest.pkl')
