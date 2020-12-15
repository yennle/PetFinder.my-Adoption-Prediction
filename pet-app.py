# from pycaret.regression import load_model, predict_model
from PIL import Image
import streamlit as st
import pandas as pd
# import numpy as np
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

# strealit update: binary to text buffer
import io

st.set_option('deprecation.showfileUploaderEncoding', False)

model = load('./model-building/pet_model.joblib')

####################
# Selection Options
####################
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
dog_breed = breed[(breed["Type"] == 1)]
cat_breed = breed[(breed["Type"] == 2)]
dog_breed_options = dict(zip(dog_breed['BreedID'], dog_breed['BreedName']))
cat_breed_options = dict(zip(cat_breed['BreedID'], cat_breed['BreedName']))
# State options
sta = pd.read_csv('./data/state_labels.csv')
state_options = dict(zip(sta['StateID'], sta['StateName']))

pet_type = 1


def user_input_features():

    # Type
    c_pt, c_q = st.beta_columns(2)
    pet_type = c_pt.selectbox('Pet Type', options=list(
        pet_options.keys()), format_func=lambda x: pet_options[x])

    load_img(pet_type)
    # Quantity
    quantity = c_q.selectbox(
        'Quantity', [x for x in range(1, 21)])
    ####################################

    # Age
    age = st.slider('Age (months)', 0, 255, 3)
    ####################################

    c_co, c_b, c_g = st.beta_columns(3)
    # Breed : show options based on type selection
    if(pet_type == 1):
        breed1 = c_b.selectbox('Breed', options=list(
            dog_breed_options.keys()), format_func=lambda x: dog_breed_options[x])
    else:
        breed1 = c_b.selectbox('Breed', options=list(
            cat_breed_options.keys()), format_func=lambda x: cat_breed_options[x])
    # Gender
    gender = c_g.selectbox('Gender', options=list(
        gender_options.keys()), format_func=lambda x: gender_options[x])
    # Color
    color1 = c_co.selectbox('Color', options=list(
        color_options.keys()), format_func=lambda x: color_options[x])
    ####################################
    c_ms, c_fl = st.beta_columns(2)
    # Maturity Size
    maturity_size = c_ms.selectbox('Maturity Size', options=list(
        maturity_size_options.keys()), format_func=lambda x: maturity_size_options[x])
    # Fur Length
    fur_length = c_fl.selectbox('Fur length', options=list(
        fur_length_options.keys()), format_func=lambda x: fur_length_options[x])
    ####################################

    c1, c2, c3 = st.beta_columns(3)
    # Vaccinated
    vaccinated = c1.selectbox('Vaccinated', options=list(
        vaccinated_options.keys()), format_func=lambda x: vaccinated_options[x])
    # Dewormed
    dewormed = c2.selectbox('Dewormed', options=list(
        dewormed_options.keys()), format_func=lambda x: dewormed_options[x])
    # Sterilized
    sterilized = c3.selectbox('Sterilized', options=list(
        dewormed_options.keys()), format_func=lambda x: dewormed_options[x])
    # Health
    health = st.selectbox('Pet Type', options=list(
        health_options.keys()), format_func=lambda x: health_options[x])
    ####################################
    c_s, c_p, c_v = st.beta_columns(3)
    # State
    state = c_s.selectbox('State', options=list(
        state_options.keys()), format_func=lambda x: state_options[x])
    # Photo Amount
    photo_amt = c_p.selectbox('Photo Amount', [x for x in range(0, 31)])
    # Video Amount
    video_amt = c_v.selectbox('Video Amount', [x for x in range(0, 9)])
    ####################################
    # Fee
    fee = st.slider('Fee', 0, 400, 0)

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
            'VideoAmt': video_amt,
            'PhotoAmt': photo_amt
            }
    features = pd.DataFrame(data, index=[0])
    return features


def predict(model, input_df):
    predictions = model.predict(input_df)
    return predictions


def load_img(pet_type):
    print(pet_type)
    if (pet_type == 1):
        return st.sidebar.image(Image.open('./img/dog.png'))
    else:
        return st.sidebar.image(Image.open('./img/cat.png'))


def run():

    # image = Image.open('l/imgogo.png')
    img = ""
    # st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))

    # st.sidebar.success('https://github.com/memoatwit/dsexample/')

    st.image(Image.open('./img/logo.png'))
    st.title("Pet Adoption Prediction App")
    st.info("In this application, we will be developing algorithms to predict the **adoptability of pets** - specifically, how quickly is a pet adopted? The AI tools that will guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.")

####################
# Online
####################
    if add_selectbox == 'Online':

        # age = st.number_input('Age', min_value=1, max_value=100, value=25)
        # sex = st.selectbox('Sex', ['male', 'female'])
        # bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        # children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        # if st.checkbox('Smoker'):
        #     smoker = 'yes'
        # else:
        #     smoker = 'no'
        # region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output = ""

        # input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
        input_df = user_input_features()
        print(input_df.columns)
        adoption_speed = {0: 'Pet should be adopted on the same day as it was listed.',
                          1: 'Pet should be adopted between 1 and 7 days (1st week) after being listed.',
                          2: 'Pet should be adopted between 8 and 30 days (1st month) after being listed.',
                          3: 'Pet should be adopted between 31 and 90 days (2nd & 3rd month) after being listed.',
                          4: 'Possibly, no adoption after 100 days of being listed.'
                          }
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)[0]
            output = adoption_speed[output]
            st.success('{}'.format(output))


####################
# BATCH
####################

    if add_selectbox == 'Batch':
        try:
            file_buffer = st.file_uploader(
                "Upload CSV file with 16 features", type=["csv"])
            st.markdown("""
                    [Example CSV input file](https://raw.githubusercontent.com/yennle/PetFinder.my-Adoption-Prediction/main/pet_example.csv)
                    """)
            # st.image(Image.open('./img/example.png'), use_column_width=True)
            # bytes_data = file_buffer.read()
            # s = str(bytes_data)
            # file_upload = io.StringIO(s)
            data = pd.read_csv(file_buffer)
            print(data.columns)
            predictions = predict(model=model, input_df=data)
            adoption_speed = {0: 'the same day as it was listed.',
                              1: '1 and 7 days (1st week) after being listed.',
                              2: '8 and 30 days (1st month) after being listed.',
                              3: '31 and 90 days (2nd & 3rd month) after being listed.',
                              4: 'no adoption after 100 days of being listed.'
                              }

            result = map(lambda x: adoption_speed[x], predictions)
            result_df = pd.DataFrame(list(result), columns=['Meaning'])
            prediction_df = pd.DataFrame(
                predictions, columns=['Prediction Value'])
            df = pd.concat([prediction_df, result_df], axis=1)
            st.write(df)
            # st.write(predictions)
        except:
            st.write("Please upload a valid CSV file.")


if __name__ == '__main__':
    run()
