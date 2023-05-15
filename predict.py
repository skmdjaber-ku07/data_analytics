import pandas as pd
import joblib

# Load the trained model from disk
rfc = joblib.load('models/random_forest.pkl')

# Load the new data into a pandas DataFrame
new_data = pd.DataFrame({
    'level': [2],
    'age': [35],
    'gender': ['M'],
    'income': [50000],
    'debt_ratio': [0.25]
})

# Make predictions on the new data
predictions = rfc.predict(new_data)

print('Predicted Reason:', predictions[0][0])
print('Predicted Method:', predictions[0][1])
