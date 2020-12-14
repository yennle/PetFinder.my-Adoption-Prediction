import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
pet = pd.read_csv('../data/pet_cleaned.csv')


# Ordinal feature encoding
# https://www.kaggle.com/c/petfinder-adoption-prediction
df = pet.copy()
target = 'AdoptionSpeed'
encode = ['Type', 'Breed1', 'Gender', 'Color1', 'MaturitySize',
          'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

for col in df.columns:
    print(col)
# target_mapper = {'Pet was adopted on the same day as it was listed.':0,
#                 'Pet was adopted between 1 and 7 days (1st week) after being listed.':1,
#                 'Pet was adopted between 8 and 30 days (1st month) after being listed.':2,
#                 'Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.':3,
#                 'No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).':4,
#                 }
# def target_encode(val):
#     return target_mapper[val]

# df['AdoptionSpeed'] = df['AdoptionSpeed'].apply(target_encode)

# Separating X and y
X = df.drop('AdoptionSpeed', axis=1)
Y = df['AdoptionSpeed']

# Build random forest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
pickle.dump(clf, open('pet_clf.pkl', 'wb'))
