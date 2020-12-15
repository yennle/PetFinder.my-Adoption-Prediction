# Pet-Adoption-Prediction

## Introduction

For our final project, we worked on [PetFinder.my Adoption Speed competition from Kaggle](https://www.kaggle.com/c/petfinder-adoption-prediction 'Petfinder.my Adoption Prediction Kaggle') . The objective is to predict the estimated time for new pets to be adopted based on the pet's listing on PetFinder.my. Although this project is challenging, it is meaningful. If the pet adoption speed is predicted well, more efficient resource allocation can be implemented to improve overall adoption performance and, subsequently, reduce sheltering and fostering costs.

We also developed a web application for our prediction using Streamlit and then deployed it using Heroku.

## Selection of Data

This project consists of 6 sources of data, images, metadata, and sentiment data. We primarily focused on the main file train.csv that has **14,993 samples** and **24 features**.

We first removed some irrelevant features to the adoption speed prediction: _PetID_, _RescuerID_, _Name_. We removed _Color2_, _Color3_, and _Breed2_ because there are huge portions of unknowns in each column. We also dropped the description for now to ease the process.

The data has **five numeric features**: _Age_, _Fee_, _VideoAmt_, _PhotoAmt_, and _Quantity_ and **ten categorical features**: _Type_, _Gender_, _MaturitySize_, _FurLength_, _Vaccinated_, _Dewormed_, _Sterilized_, _Health_, _Breed1_, and _Color1_.

We used **OneHotEncoder** and **ColumnTransformer** to encode the categorical features and kept the rest numeric features as is. With this transformation, the test accuracy of Logistic Regression went up about **5%**, but it did not help much for **Random Forest Classifier**.

Based on Professor's example, we created a pipeline for automating the encoding and transforming process for new data. We set missing data to -1 instead of 0 to avoid confusion. The shown values are obtained by performing a grid search over these arguments.

## Methods:

**Tools:**

- Numpy, Pandas, Matplotlib, and Seaborn for data analysis and visualization
- Scikit-learn for inference
- Streamlit for web app design
- Github for web app deployment and version control
- VS Code as IDE
- Anaconda for Juypter Notebook

**Inference methods used with Scikit:**

- Models: KNeighborsClassifier, Naive Bayes, Logistic Regression, RandomForestClassifier
- Features: OneHotEncoder/ColumnTransformer, StandardScaler, Simpleimputer for missing values
- GridSearchCV for hyperparameter tunning

## Results:

- Online: User inputs each feature manually for predicting a single adoption speed.

![Results Online](https://media.giphy.com/media/qW4gkS5kR6WBeqkXAC/giphy.gif 'Online')

- Batch: It allows the user to upload a CSV file with the 16 features for predicting many instances at once.

![Results Batch](https://media.giphy.com/media/GqQXEuahbUNWBz9uZa/giphy.gif 'Batch')

## Discussion:

After exploring various classification algorithms, we found out that _Random Forest Classification_ gives the best accuracy. Hyperparameter tuning with _GridSearchCV_ significantly improved accuracy. Therefore, we decided the deploy the **_GridSearchCV Random Forest model_**. The data was split **75% for training** and **25% for testing** and had a **test accuracy of 70%.**

Each model's accuracy scores also suggest that our models are insufficient for the prediction task as they do not incorporate the information from the profile images or the animal descriptions (text data). Without visual information extracted from animal photos and textual information extracted from the descriptions, the models cannot predict the adoption rate as accurately as we expected.

Going through some top-rated notebooks for competition on Kaggle, we noticed that this data could also be interpreted using regression algorithms. There are some other popular algorithms, such as LightGBM, Naural Networks, and XGBoost, which we did not have a chance to explore due to time constraints.

## Summary

This Pet Prediction project deploys a supervised classification model to predict adoption speed based on 16 features: Type, Age, Breed, Gender, Color, Maturity Size, Fur Length, Vaccinated, Dewormed, Sterilized, Health, Quantity, Fee, State,Video Amount, Photo Amount

After experimenting with various feature engineering techniques, the deployed model's testing accuracy hovers around 70%.
The web app is designed using Streamlit, and can do online and batch processing, and is deployed using Heroku and Streamlit. The Heroku app is live at https://ds-example.herokuapp.com/.
