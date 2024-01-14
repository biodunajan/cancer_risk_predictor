# Machine Learning Cancer Risk  Prediction Web App Using Python & Streamlit

# Import the necessary libraries and dependencies
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

# define the layout and container for widget
PAGE_CONFIG = {"page_title": "Cancer Risk Prediction System",
               "page_icon": "✅", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)

# define the column layout for the web application
col1, col2, col3 = st.columns(3)

# define the web app welcome/landing page and get input from user

with st.sidebar:
    # Display logo for the landing page
    image = Image.open('best.jpg')
    new_image = image.resize((1000, 450))
    st.image(new_image, use_column_width=True)

    st.subheader("Do you know?")
    st.write("Everyone has a certain risk of developing cancer. ")
    st.write(
        "A combination of genes, lifestyle, and environment can affect this risk. ")
    st.write("Want to know your Cancer Risk Level?")
    st.write("Please complete the short questionnaire on the right side of the app and click the submit button to see your result. ")

with st.container():
    def get_user_input():

        st.subheader("Please complete the following questions to know your risk level and how you compare "
                     "to normal.")

        gender = st.radio('● Gender: ',
                          ('Female', 'Male', 'Prefer not to say'))
        if gender == 'Female':
            gender = 0
        elif gender == 'Female':
            gender = 1
        else:
            gender = 2

        freq_exercise = st.slider('● Frequency of exercise: '
                                  'How often do you exercise or involve in physical activities in a week?', min_value=0,
                                  max_value=10, value=0, step=1)

        sugar_intake = st.slider('● Sugar intake: '
                                 'How often do you have sugary drinks and processed food high '
                                 'in fat and sugar each week?',
                                 min_value=0, max_value=10, value=0, step=1)

        alcohol = st.slider('● Alcohol intake: '
                                  'How would you define your alcohol consumption per day? '
                                  'If you do not take alcohol, please leave slider at 0. '
                                  'One unit of alcohol equals 10ml or 8g of pure alcohol.  '
                                  'This is equivalent of 1 small glass of wine, beer or spirits. ',
                                  min_value=0, max_value=10, value=0, step=1)

        smoke_ppy = st.slider('● Smokes per pack year(sppy): '
                              'How would you rate your smokes per pack-year? '
                              'If you do not smoke, please leave slider at 0. '
                              'sppy is a numerical value of lifetime tobacco exposure. '
                              'A pack year is defined as number of packs of cigarettes a person has smoked '
                              'every day multiplied by the number of years smoked. ',
                              min_value=0, max_value=100, value=0, step=1)

        freq_cold = st.slider('● Personal or family history of chronic diseases: '
                              'Do you have any of the following chronic illness, allergies, or symptoms? '
                              'frequent cold or urination; pain or bloody discharge from any part '
                              'of the body; '
                              'loss of weigh or appetite; shortness of breath; diabetes; '
                              'lung or kidney problem; fatigue amongst others. '
                              'If yes, please rate the pain or discomfort level. '
                              ' If no, please leave slider at 0.',
                              min_value=0, max_value=10, value=0, step=1)

        sun_exposure = st.slider('● Exposure to sun: '
                                 'How would you rate your level of exposure to sun in your location '
                                 'between 11am and 3pm?',
                                 min_value=0, max_value=10, value=0, step=1)

        co2_emission = st.slider('● Exposure to co2 emission: '
                                 'How would you rate your level of exposure to pollution '
                                 'resulting co2 emission from automobiles ?'
                                 'If not applicable, please leave slider at 0 ',
                                 min_value=0, max_value=10, value=0, step=1)

        industrial_pollution = st.slider('● Exposure to industrial pollution: '
                                         'How would you rate your level of exposure to pollution '
                                         'resulting from industrial practices such as: burning coal, '
                                         'fossil fuel like oil, natural gas, and petroleum; chemical solvents; '
                                         'untreated gas and liquid; improper disposal of radioactive '
                                         'materials ? '
                                         'If not applicable, please leave slider at 0',
                                         min_value=0, max_value=10, value=0, step=1)

        domestic_pollution = st.slider('● Exposure to domestic pollution: '
                                       'How would you rate your level of exposure to pollution '
                                       'resulting from sewage and waste from foods preparation, garbage,'
                                       'dishwashing, toilets, baths, sinks etc?'
                                       'If not applicable, please leave slider at 0',
                                       min_value=0, max_value=10, value=0, step=1)

        user_data = {'gender': gender,
                     'freq_exercise': freq_exercise,
                     'sugar_intake': sugar_intake,
                     'alcohol': alcohol,
                     'smoke_ppy': smoke_ppy,
                     'freq_cold': freq_cold,
                     'sun_exposure': sun_exposure,
                     'co2_emission': co2_emission,
                     'industrial_pollution': industrial_pollution,
                     'domestic_pollution': domestic_pollution}

        features = pd.DataFrame(user_data, index=[0])

        return features

    user_input = get_user_input()

# Loading the machine learning trained model and predicting based on the user_input
with st.container():

    def interactive_chart():
        global keep
        st.subheader("How you compare to Normal")
        df = pd.read_csv('app_user_data.csv')
        df = df.append(user_input, ignore_index=False)
        df_1 = df.drop('gender', axis=1)
        last = df_1.iloc[-1].tolist()
        keep = [i for i in last]
        normal = pd.read_excel('Normal Answer.xlsx')
        threshold = normal['Normal'][0:10].tolist()
        column = df_1.columns.tolist()

        fig = plt.figure(figsize=(18, 8))
        X_axis = np.arange(len(column))

        plt.bar(X_axis - 0.2, last, 0.4, label='User', color='cyan')
        plt.bar(X_axis + 0.2, threshold, 0.4, label='Normal', color='green')

        plt.xticks(X_axis, column)
        plt.legend()
        plt.show()
        st.pyplot(fig)

    def low_risk():
        st.subheader("Recommendation")
        st.write(
            "Awesome! Your numbers  show you are right on the border of normal risk profile. However, you can maintain a low risk level by doing the following: ")
        st.write("● Exercise regularly. Being active can help you to lose or keep a healthy weight, and this can lower your risk of 13 different types of cancer.")
        st.write("● Cut down, and if possible avoid tobacco use, including cigarettes and smokeless tobacco. Smoking increases the risk of at least 15 different types of cancer.")
        st.write("● Limit alcohol use.  The less you drink, the lower the risk of cancer. The government guideline is no more than around 7 drinks per week (14 units of alcohol).")
        st.write("● Reduce exposure to ultraviolet radiation (e.g sun). Getting sunburnt just once every 2 years can triple the risk of melanoma skin cancer.")
        st.write("● Reduce exposure to co2 emission from automobiles and industrial activities")
        st.write("● Avoid indoor smoke from household use of solid fuels.")
        st.write("● Eat a healthy diet with plenty of fruit and vegetables.")
        st.write("● Limit intake of processed/red meat including hog dogs, pepperoni, chorizo, salami, ham, and bacon.")
        st.write("● Limit sugary drinks and processed food high in fat and sugar.")
        st.write("● Get regular medical care or see a doctor If you are having chronic infections, diseases, symptoms or discomfort such as (frequent cold, high blood pressure, bloody discharge or  pain in any part of the body etc).")

    def high_risk():
        st.subheader("Recommendation")
        st.write("Based on your comparision to normal, you can reduce your cancer  risk level by modifying or avoiding the following risk factors.")
        st.write("● Cut down, and if possible avoid tobacco use, including cigarettes and smokeless tobacco. Smoking increases the risk of at least 15 different types of cancer.")
        st.write("● Eat a healthy diet with plenty of fruit and vegetables.")
        st.write("● Exercise regularly. Being active can help you to lose or keep a healthy weight, and this can lower your risk of 13 different types of cancer.")
        st.write("● Limit alcohol use.  The less you drink, the lower the risk of cancer. The government guideline is: no more than around 7 drinks per week (14 units of alcohol).")
        st.write("● Reduce exposure to ultraviolet radiation (e.g sun). Getting sunburnt just once every 2 years can triple the risk of melanoma skin cancer.")
        st.write("● Avoid urban air pollution and indoor smoke from household use of solid fuels.")
        st.write("● Get vaccinated against hepatitis B and human papillomavirus (HPV).")
        st.write("● Limit intake of processed/red meat including hog dogs, pepperoni, chorizo, salami, ham, and bacon.")
        st.write("● Get regular medical care or see a doctor If you are having chronic infections, diseases, symptoms or discomfort such as: frequent cold, high blood pressure, bloody discharge or  pain in any part of the body etc.")

    def load_model():
        with st.form('Form1', clear_on_submit=False):
            st.subheader("Your Result")
            # Loading the saved model
            rf_model = joblib.load('final_rfmodel.joblib')
            submitted = st.form_submit_button('Submit to Predict')
            if submitted:
                prediction = rf_model.predict(user_input)
                if prediction == 1:
                    st.metric(label='Your Cancer Risk Level', value=prediction, delta="High: Positive",
                              delta_color="inverse")
                    interactive_chart()
                    high_risk()
                elif prediction == 0:
                    st.metric(label='Your Cancer Risk Level', value=prediction, delta="Low: Negative",
                              delta_color="normal")

                    interactive_chart()
                    low_risk()

                df = pd.read_csv('app_user_data.csv')
                df = df.append(user_input, ignore_index=False)
                df.to_csv('app_user_data.csv', index=False)

    load_model()
