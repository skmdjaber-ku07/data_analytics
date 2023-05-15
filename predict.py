import pandas as pd
import joblib

rfc = joblib.load('random_forest.pkl')

new_data = pd.DataFrame({
    'level': [2],
    'age': [35],
    'gender': ['M'],
    'income': [50000],
    'debt_ratio': [0.25]
})

predictions = rfc.predict(new_data)

print('Predicted Reason:', predictions[0][0])
print('Predicted Method:', predictions[0][1])
