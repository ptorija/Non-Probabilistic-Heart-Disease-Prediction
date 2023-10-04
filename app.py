import pandas as pd

# Load the data
data = pd.read_csv('data/heart_2020_cleaned.csv')


# Transform the boolean columns into 0,1
data['HeartDisease'] = data['HeartDisease'].replace({'Yes': 1, 'No': 0})

data['Smoking'] = data['Smoking'].replace({'Yes': 1, 'No': 0})

data['AlcoholDrinking'] = data['AlcoholDrinking'].replace({'Yes': 1, 'No': 0})

data['Stroke'] = data['Stroke'].replace({'Yes': 1, 'No': 0})

data['DiffWalking'] = data['DiffWalking'].replace({'Yes': 1, 'No': 0})
sex_mapping = {
    'Male': 0,
    'Female': 1
}
data['Sex'] = data['Sex'].map(sex_mapping)

age_mapping = {
    '18-24': 21,
    '25-29': 27,
    '30-34': 32,
    '35-39': 37,
    '40-44': 42,
    '45-49': 47,
    '50-54': 52,
    '55-59': 57,
    '60-64': 62,
    '65-69': 67,
    '70-74': 72,
    '75-79': 77,
    '80 or older': 85
}
data['AgeCategory'] = data['AgeCategory'].map(age_mapping)

data = data[data['Race'] != 'American Indian/Alaskan Native']
data = pd.get_dummies(data, columns=['Race'], prefix=['Race'])

data = data[data['Diabetic'].isin(['Yes', 'No'])]
data['Diabetic'] = data['Diabetic'].replace({'Yes': 1, 'No': 0})

data['PhysicalActivity'] = data['PhysicalActivity'].replace({'Yes': 1, 'No': 0})

health_mapping = {
    'Excellent': 5,
    'Very good': 4,
    'Good': 3,
    'Fair': 2,
    'Poor': 1
}
data['GenHealth'] = data['GenHealth'].map(health_mapping)

data['Asthma'] = data['Asthma'].replace({'Yes': 1, 'No': 0})

data['KidneyDisease'] = data['KidneyDisease'].replace({'Yes': 1, 'No': 0})

data['SkinCancer'] = data['SkinCancer'].replace({'Yes': 1, 'No': 0})

print(data.info())
print(data.head())