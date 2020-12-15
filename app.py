from joblib import load
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Selection options
pet_options = {1: "Dog", 2: "Cat"}
gender_options = {1: "Male", 2: "Female", 3: "Mixed"}
maturity_size_options = {1: 'Small', 2: 'Medium',
                         3: 'Large', 4: 'Extra Large'}
color_options = {1: 'Black', 2: 'Brown', 3: 'Golden',
                 4: 'Yellow', 5: 'Cream', 6: 'Gray', 7: 'White'}
fur_length_options = {1: 'Short', 2: 'Medium', 3: 'Long'}
vaccinated_options = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
dewormed_options = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
sterilized_options = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
health_options = {1: 'Healthy', 2: 'Minor Injury',
                  3: 'Serious Injury'}
# Breed options
breed = pd.read_csv('./data/breed_labels.csv')

# pet_data = pd.read_csv('./data/pet_cleaned.csv')
# breed_data = pet_data[['Breed1', 'Type']]
# breed_data.columns = ['BreedID', 'Type']
# print(breed_data)
# final_breed = pd.merge(breed_data, breed, how='right', on=['BreedID', 'Type'])
# print(final_breed)
# breed = breed[(breed['BreedID'] == pet_data['Breed1'])
#               & (breed['Type'] == pet_data['Type'])]
# print(breed)
dog_breed = breed[(breed["Type"] == 1)]
cat_breed = breed[(breed["Type"] == 2)]
dog_breed_options = dict(zip(dog_breed['BreedID'], dog_breed['BreedName']))
cat_breed_options = dict(zip(cat_breed['BreedID'], cat_breed['BreedName']))

# State options
sta = pd.read_csv('./data/state_labels.csv')
state_options = dict(zip(sta['StateID'], sta['StateName']))


##############################
# SIDEBAR : Input Processing
##############################
st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/yennle/PetFinder.my-Adoption-Prediction/main/pet_example.csv)
""")
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader(
    "Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # Type
        pet_type = st.sidebar.selectbox('Pet Type', options=list(
            pet_options.keys()), format_func=lambda x: pet_options[x])
        # Age
        age = st.sidebar.slider('Age (months)', 0, 255, 3)
        # Breed : show options based on type selection
        if(pet_type == 1):
            breed1 = st.sidebar.selectbox('Breed', options=list(
                dog_breed_options.keys()), format_func=lambda x: dog_breed_options[x])
        else:
            breed1 = st.sidebar.selectbox('Breed', options=list(
                cat_breed_options.keys()), format_func=lambda x: cat_breed_options[x])
        # Gender
        gender = st.sidebar.selectbox('Gender', options=list(
            gender_options.keys()), format_func=lambda x: gender_options[x])
        # Color
        color1 = st.sidebar.selectbox('Color', options=list(
            color_options.keys()), format_func=lambda x: color_options[x])
        # Maturity Size
        maturity_size = st.sidebar.selectbox('Maturity Size', options=list(
            maturity_size_options.keys()), format_func=lambda x: maturity_size_options[x])
        # Fur Length
        fur_length = st.sidebar.selectbox('Fur length', options=list(
            fur_length_options.keys()), format_func=lambda x: fur_length_options[x])
        # Vaccinated
        vaccinated = st.sidebar.selectbox('Vaccinated', options=list(
            vaccinated_options.keys()), format_func=lambda x: vaccinated_options[x])
        # Dewormed
        dewormed = st.sidebar.selectbox('Dewormed', options=list(
            dewormed_options.keys()), format_func=lambda x: dewormed_options[x])
        # Sterilized
        sterilized = st.sidebar.selectbox('Sterilized', options=list(
            dewormed_options.keys()), format_func=lambda x: dewormed_options[x])
        # Health
        health = st.sidebar.selectbox('Pet Type', options=list(
            health_options.keys()), format_func=lambda x: health_options[x])
        # Quantity
        quantity = st.sidebar.slider('Quantity', 1, 20, 1)
        # Fee
        fee = st.sidebar.slider('Fee', 0, 400, 0)
        # State
        state = st.sidebar.selectbox('State', options=list(
            state_options.keys()), format_func=lambda x: state_options[x])
        # Photo Amount
        photo_amt = st.sidebar.slider('Photo Amount', 0, 30, 2)
        # Video Amount
        video_amt = st.sidebar.slider('Video Amount', 0, 30, 2)

        # Generate dict data
        data = {'Type': pet_type,
                'Age': age,
                'Breed1': breed1,
                'Gender': gender,
                'Color1': color1,
                'MaturitySize': maturity_size,
                'FurLength': fur_length,
                'Vaccinated': vaccinated,
                'Dewormed': dewormed,
                'Sterilized': sterilized,
                'Health': health,
                'Quantity': quantity,
                'Fee': fee,
                'State': state,
                'PhotoAmt': photo_amt,
                'VideoAmt': video_amt

                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

##############################
# SIDEBAR : Prediction Processing
##############################

# Combines user input features with entire pet dataset
# This will be useful for the encoding phase
# pet_raw = pd.read_csv('./data/pet_cleaned.csv')
# pet = pet_raw.drop(columns=['AdoptionSpeed'])
# df = pd.concat([input_df, pet], axis=0)

# # Encoding of ordinal features
# encode = ['Type', 'Breed1', 'Gender', 'Color1', 'MaturitySize',
#           'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df, dummy], axis=1)
#     del df[col]


# if uploaded_file is not None:
#     df = df[:len(input_df)]
# else:
#     df = df[:1]  # Selects only the first row (the user input data)

# Reads in saved classification model
# load_clf = pickle.load(open('./model-building/pet_model.joblib', 'rb'))
model = load('./model-building/pet_model.joblib')
# Apply model to make predictions


def predict(model, input_df):
    # predictions_df = predict_model(estimator=model, data=input_df)
    predictions = model.predict(input_df)
    # predictions = predictions_df['Label'][0]
    # get rid of unfeasible predictions
    return predictions


print(input_df.columns)
# prediction = load_clf.predict(input_df)
# print(predict(model, input_df))
# print('test prediction')
# for i in prediction:
#     print(i)
################################
# DISPLAY
################################


# Load style.css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)


# Introduction
st.write("""
# Pet Adoption Prediction App

In this application, we will be developing algorithms to predict the **adoptability of pets** - specifically, how quickly is a pet adopted? 
The AI tools that will guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.

Data obtained from the [petfinder-adoption-prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) .
""")

# Displays the user input features
st.header('''
**User Input features**
''')
# Define the input source


def format_data(data, title, column):
    with column:
        st.subheader(title+':')
        t = "<div class='highlight blue bold'>" + data+"</div>"
        st.markdown(t, unsafe_allow_html=True)


adoption_speed = {0: 'Pet should be adopted on the same day as it was listed.',
                  1: 'Pet should be adopted between 1 and 7 days (1st week) after being listed.',
                  2: 'Pet should be adopted between 8 and 30 days (1st month) after being listed.',
                  3: 'Pet should be adopted between 31 and 90 days (2nd & 3rd month) after being listed.',
                  4: 'Possibly, no adoption after 100 days of being listed.'
                  }
if uploaded_file is not None:
    st.write('Input data is imported from CSV file')
    adoption_speed_1 = {0: 'the same day as it was listed.',
                        1: '1 and 7 days (1st week) after being listed.',
                        2: '8 and 30 days (1st month) after being listed.',
                        3: '31 and 90 days (2nd & 3rd month) after being listed.',
                        4: 'no adoption after 100 days of being listed.'
                        }
    result = map(lambda x: adoption_speed_1[x], prediction)
    result_df = pd.DataFrame(list(result), columns=['Meaning'])
    prediction_df = pd.DataFrame(prediction, columns=['Prediction Value'])
    df = pd.concat([prediction_df, result_df], axis=1)
    st.write(df)
else:
    st.write("Input data is defined from the user's entry ")
    local_css("style.css")
    # First row: Pet type, Breed, Gender
    c1, c2, c4 = st.beta_columns(3)
    format_data(pet_options[input_df.iloc[0]["Type"]], 'Pet type', c1)

    if(pet_options[input_df.iloc[0]["Type"]] == 'Dog'):
        format_data(dog_breed_options[input_df.iloc[0]["Breed1"]], 'Breed', c2)
    else:
        format_data(cat_breed_options[input_df.iloc[0]["Breed1"]], 'Breed', c2)
    format_data(gender_options[input_df.iloc[0]["Gender"]], 'Gender', c4)
    # Second row: Age, Color, Maturity size, Fur length
    c3, c5, c6, c7 = st.beta_columns(4)
    format_data(str(input_df.iloc[0]["Age"]), 'Age', c3)
    format_data(color_options[input_df.iloc[0]["Color1"]], 'Color', c5)
    format_data(maturity_size_options[input_df.iloc[0]
                                      ["MaturitySize"]], 'Maturity size', c6)
    format_data(fur_length_options[input_df.iloc[0]
                                   ["FurLength"]], 'Fur length', c7)

    # Third row: Vaccinated, Dewormed, Sterilized
    c8, c9, c10 = st.beta_columns(3)
    format_data(vaccinated_options[input_df.iloc[0]
                                   ["Vaccinated"]], 'Vaccinated', c8)
    format_data(dewormed_options[input_df.iloc[0]["Dewormed"]], 'Dewormed', c9)
    format_data(sterilized_options[input_df.iloc[0]
                                   ["Sterilized"]], 'Sterilized', c10)

    # Forth row: Quanity, Fee, State, Photo
    c11, c12, c13, c14 = st.beta_columns(4)
    format_data(str(input_df.iloc[0]["Quantity"]), 'Quantity', c11)
    format_data(str(input_df.iloc[0]["Fee"]), 'Fee', c12)
    format_data(state_options[input_df.iloc[0]["State"]], 'State', c13)
    format_data(str(input_df.iloc[0]["PhotoAmt"]), 'Photo Amount', c14)

    st.write('''
    Prediction
    ======
    ''')

    st.success(adoption_speed[prediction[0]])

# if uploaded_file is not None:
#     st.write(prediction)
# else:


# st.subheader('Prediction Probability')
# st.write(prediction_proba)
