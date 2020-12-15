from joblib import dump, load
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import pandas as pd
train = pd.read_csv('../train.csv')
drop = ['Name', 'Breed2', 'Color2', 'Color3',
        'RescuerID', 'PetID', 'Description']
train = train.drop(drop, axis=1)

X = train.drop(['AdoptionSpeed'], axis=1)
y = train.AdoptionSpeed

# Train-Test Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.25, random_state=0)

# encode categorical data and create pipeline for automating the process for new data
numeric_features = ['Age', 'Fee', 'VideoAmt', 'PhotoAmt', 'Quantity']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Type', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                        'Sterilized', 'Health', 'Breed1', 'Color1']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
model_randomforest = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier',
                                      RandomForestClassifier(
                                          n_jobs=-1, n_estimators=150)
                                      )])
# model_randomforest = RandomForestClassifier(n_estimators=150)
model_randomforest.fit(Xtrain, ytrain)


# make predictions on test data
y_model_randomforest = model_randomforest.predict(Xtest)

# accuracy score
train_accuracy_rf = model_randomforest.score(Xtrain, ytrain)
test_accuracy_rf = model_randomforest.score(Xtest, ytest)
print("Random Forest train accuracy: {:.2f}%".format(train_accuracy_rf * 100))
print("Random Forest test accuracy: {:.2f}%".format(test_accuracy_rf * 100))

# then save model:
#pickle.dump(model_randomforest, open('pet_clf.pkl', 'wb'))
dump(model_randomforest, 'pet_model.joblib')
