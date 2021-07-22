import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import Binarizer

st.title("Early Diabetes Disease preiction.")


def app():
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

        with open("RF.pkl", 'rb') as pfile:  
            model_loaded=pickle.load(pfile)
        y_predicted=model_loaded.predict(prediction_df)


        if (y_predicted[0]==1): 
            st.write("Sorry to say that you are Diabetic. Probability of being Positive with Diabetes is shown in column 1.:")
        else:
            st.write("Congratulations! You are not Diabetic. Probability of being Negative with Diabetes is shown in column 0.")
        st.write(model_loaded.predict_proba(prediction_df))

app()