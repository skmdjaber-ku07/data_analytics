import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('data/dataset.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[['level','age','gender','income','debt_ratio']], data[['reason', 'method']], test_size=0.2)

# Train a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(rfc, 'models/random_forest.pkl')
