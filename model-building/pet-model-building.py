import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import pandas as pd
pet = pd.read_csv('./data/pet_cleaned.csv')


# Ordinal feature encoding
# https://www.kaggle.com/c/petfinder-adoption-prediction
df = pet.copy()
#df = pet.sample(frac=0.8, random_state=2)
target = 'AdoptionSpeed'
encode = ['Type', 'Breed1', 'Gender', 'Color1', 'MaturitySize',
          'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
print(len(df.columns))
# for col in df.columns:
#     print(col)
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
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=test_size, random_state=seed)

# Build random forest model
clf = RandomForestClassifier()
#clf.fit(X, Y)
clf.fit(X_train, Y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
# Saving the model
# pickle.dump(clf, open('pet_clf.pkl', 'wb'))
