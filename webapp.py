import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open("trained_model.sav", "rb"))


def predict_diabetes(user_input):
    # input = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
    user_input = np.asarray(user_input)
    user_input = user_input.reshape(1, -1)

    # standardize data
    # scaled_input = scaler.transform(input)

    # make prediction
    prediction = loaded_model.predict(user_input)
    print(prediction)

    if prediction[0] == 0:
        return "You're not diabetic."
    else:
        return "You're diabetic."


def main():
    # setting title for our webpage
    st.title("Do you have diabetes?")

    # getting data from user
    pregnancies = st.text_input("What's your pregnancy count?")
    glucose = st.text_input("Your glucose level?")
    bloodpressure = st.text_input("Your blood pressure?")
    skinthickness = st.text_input("Your skin thickness?")
    insulin = st.text_input("Your insulin level?")
    bmi = st.text_input("Your bmi")
    diabetespedigreefunction = st.text_input("Your diabetes pedigree function?")
    age = st.text_input("Your age?")

    # code for prediction
    diagnosis = ""

    # code for button
    if st.button("diabetes_test_result"):
        diagnosis = predict_diabetes([pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age])

    st.success(diagnosis)


if __name__ == "__main__":
    main()


print("code run successfully")
