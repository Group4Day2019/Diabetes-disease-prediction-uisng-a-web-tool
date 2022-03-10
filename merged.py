import streamlit as st
import pandas as pd
import numpy as np
import pickle

def pima():
    pickle_in = open('SVM_model.pkl', 'rb')
    classifier = pickle.load(pickle_in)

    st.markdown("## Pima Indian Diabetes Disease preiction.")
    
    st.markdown('#### Diabetes Prediction(Only for females above 21years of Age)')
    name = st.text_input("Name:")
    pregnancy = st.number_input("No. of times pregnant:", min_value=0, max_value=15, step=1)
    glucose = st.number_input("Plasma Glucose Concentration :", min_value=40, max_value=250, step=1)
    bp =  st.number_input("Diastolic blood pressure (mm Hg):", min_value=20, max_value=140, step=1)
    skin = st.number_input("Triceps skin fold thickness (mm):", min_value=5, max_value=80, step=1)
    insulin = st.number_input("2-Hour serum insulin (mu U/ml):", min_value=0, max_value=1000, step=1)
    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):", min_value=10, max_value=100, step=1)
    dpf = st.number_input("Diabetes Pedigree Function:", min_value=0.0, max_value=2.5, step=1.0)
    age = st.number_input("Age:", min_value=10, max_value=120, step=1)
    submit = st.button('Predict')

    if submit:
            prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
            if prediction == 0:
                st.write('Congratulation',name,'You are not diabetic')
            else:
                st.write(name," we are really sorry to say but it seems like you are Diabetic.")
# ...............................................................
def app():
    st.markdown("## Early Diabetes Disease preiction.")

    option1 = st.selectbox("Age(from 20 years to 65 years.), If above 35years select 'Yes' otherwise select 'No'.", ('Yes', 'No'))
    
    option2 = st.selectbox('Gender',('Male', 'Female'))
    
    option3 = st.selectbox('Polyuria',('Yes', 'No'))
    
    option4 = st.selectbox('Polydispia',('Yes', 'No'))
    
    option5 = st.selectbox('Sudden weight loss',('Yes', 'No'))

    option6 = st.selectbox('Weakness',('Yes', 'No'))

    option7 = st.selectbox('Polyphagia',('Yes', 'No'))

    option8 = st.selectbox('Genital thrush',('Yes', 'No'))

    option9 = st.selectbox('Visual blurring',('Yes', 'No'))

    option10 = st.selectbox('Itching',('Yes', 'No'))

    option11 = st.selectbox('Irritability',('Yes', 'No'))

    option12 = st.selectbox('Delayed healing',('Yes', 'No'))

    option13 = st.selectbox('Partial Paresis',('Yes', 'No'))

    option14 = st.selectbox('Muscle stiffness',('Yes', 'No'))

    option15 = st.selectbox('Alopecia',('Yes', 'No'))

    option16 = st.selectbox('Obesity',('Yes', 'No'))
    

    if st.button('Predict.'):
        lookup_dict={"Yes":1,"No":0, "Male":1, "Female":0}
        dict = {'age':[lookup_dict[option1]],
            'gender':[lookup_dict[option2]],
            'polyuria':[lookup_dict[option3]],
            "polydispia":[lookup_dict[option4]],
            "sudden":[lookup_dict[option5]],
            'weak':[lookup_dict[option6]],
            'polyghagia':[lookup_dict[option7]],
            'genital':[lookup_dict[option8]],
            'visual':[lookup_dict[option9]],
            'itch':[lookup_dict[option10]],
            'irit':[lookup_dict[option11]],
            'delay':[lookup_dict[option12]],
            'partial':[lookup_dict[option13]],
            'muscle':[lookup_dict[option14]],
            'alopecia':[lookup_dict[option15]],
            'obesity':[lookup_dict[option16]],
           }
        prediction_df = pd.DataFrame(dict)

        st.write("User details for prediction")

        st.write(prediction_df)

        with open("ETC.pkl", 'rb') as pfile:  
            model_loaded=pickle.load(pfile)
        y_predicted=model_loaded.predict(prediction_df)


        if (y_predicted[0]==1): 
            st.write("Sorry to say that you are Diabetic. Probability of being Positive with Diabetes is shown in column 1.:")
        else:
            st.write("Congratulations! You are not Diabetic. Probability of being Negative with Diabetes is shown in column 0.")
        st.write(model_loaded.predict_proba(prediction_df))


st.title("Diabetes Disease preiction app.")

st.write('Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy. Sometimes your body doesn’t make enough—or any—insulin or doesn’t use insulin well. Glucose then stays in your blood and doesn’t reach your cells.')

st.markdown('### Symptoms include the following;')
st.markdown('##### Urinating often.')
st.markdown('##### Feeling very thirsty.')
st.markdown('##### Extreme fatigue.')
st.markdown('##### Blurry vision.')


st.markdown('### Kindly choose the mode you wish to get predictions with.')
option = st.selectbox('choose',('Pima Indians', 'Early Diabetes Disease'))


if option == 'Pima Indians':
    pima()
elif option == 'Early Diabetes Disease':
    app()


