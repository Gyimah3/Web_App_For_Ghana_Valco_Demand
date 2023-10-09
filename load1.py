# Importing required Library
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
from sklearn import preprocessing
from catboost import  CatBoostRegressor
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from datetime import datetime, date as dt_date
from datetime import datetime, timedelta
import plotly.express as px
#import threadpoolctl 



# Setting up page configuration and directory path
st.set_page_config(
    page_title="Load Forecasting App",
    page_icon="🐞",
    layout="centered",
    initial_sidebar_state="collapsed",
   
)
DIRPATH = os.path.dirname(os.path.realpath(__file__))


# Setting background image

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-color:black;
background-image:
radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px),
radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px),
radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 40px),
radial-gradient(rgba(255,255,255,.4), rgba(255,255,255,.1) 2px, transparent 30px);
background-size: 550px 550px, 350px 350px, 250px 250px, 150px 150px;
background-position: 0 0, 40px 60px, 130px 270px, 70px 100px;
}
</style>
'''
st.markdown(page_bg_img,unsafe_allow_html=True)



# Setting up logo
left1, left2, mid,right1, right2 = st.columns(5)
with left1:
    #image1= Image.open(r"C:\\Users\\USER\Desktop\\lo.jpg")
    st.image('https://amvionlabs.com/wp-content/uploads/2020/12/demand-forecast-1.png', width=400,caption=None, use_column_width=None, clamp=100, channels="RGB", output_format='png')
with right1:
    #image= Image.open(r"C:\\Users\\USER\Desktop\\loi.jpg")
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTL0Zziaqx1ZTVsOXXTYmyr_Q7fVKLPT4SvPw&usqp=CAU',caption=None, use_column_width=None, clamp=100, channels="RGB", output_format='png', width=317,)



# Setting up Sidebar
social_acc = ['Data Field Description', 'EDA', 'About App']
social_acc_nav = st.sidebar.radio('*INFORMATION SECTION*', social_acc)

if social_acc_nav == 'Data Field Description':
    st.sidebar.markdown("<h2 style='text-align: center;'> Data Field Description </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown("**Date:** The date you want to predict Load for")
    st.sidebar.markdown("**Thermal:** Thermal Power generated")
    st.sidebar.markdown("**Hydro:** Hydro Power Gnerated")
    st.sidebar.markdown("**Import:** Power imported")
    st.sidebar.markdown("**Export:** Power exported from Generator")
    st.sidebar.markdown("**Weather Remperature:** The temperature of the weather in °C")
    st.sidebar.markdown("**Weather Humidity:** The amount of water vapour the air is holdimg in %")
    st.sidebar.markdown("**Wind Speed:** The speed of the wind in km/h")
    st.sidebar.markdown("**Weather Pressure:** The Atmospheric pressure in hpa")

elif social_acc_nav == 'EDA':
    st.sidebar.markdown("<h2 style='text-align: center;'> Exploratory Data Analysis </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''The exploratory data analysis of this project can be find in a Jupyter notebook from the link below''')
    st.sidebar.markdown("[Open Notebook](https://github.com/Kyei-frank/Regression-Project-Store-Sales--Time-Series-Forecasting/blob/main/project_workflow.ipynb)")

elif social_acc_nav == 'About App':
    st.sidebar.markdown("<h2 style='text-align: center;'> Load Forecasting App </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown("This App predicts the Load Demand for Volta Aluminium Company using XGBoost model.")
    st.sidebar.markdown("")
    st.sidebar.markdown("[ Visit Github Repository for more information](https://github.com/Kyei-frank/Regression-Project-Store-Sales--Time-Series-Forecasting)")
    st.sidebar.markdown("Dedicated to Ing. Isaac Otchere❄️.")
    st.sidebar.markdown("Contact Developers")
    st.sidebar.markdown("[ Gyimah Gideon](https://wa.link/yvcpjd)")
    st.sidebar.markdown("[ Antwi Augustine](https://wa.link/e7w16w)")
    st.sidebar.markdown("[ Damba Eric](https://wa.link/c648pg)")
    st.sidebar.markdown("")
    

# Loading Machine Learning Objects
@st.cache_data()
def Load_ml_items(relative_path):
    "Load ML items to reuse them"
    with open(relative_path, 'rb') as file:
        loaded_object_4 = pickle.load(file)
    return loaded_object_4

loaded_object_4 = Load_ml_items('ml_items_33')
model, Data = loaded_object_4['model'], loaded_object_4['Data']



# Extracting year, month, day and week,etc and making new column
def getDateFeatures(df, date):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.isocalendar().week
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df['is_weekend']= np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start']= df.date.dt.is_month_start.astype(int)
    df['is_month_end']= df.date.dt.is_month_end.astype(int)
    df['quarter']= df.date.dt.quarter
    df['is_quarter_start']= df.date.dt.is_quarter_start.astype(int)
    df['is_quarter_end']= df.date.dt.is_quarter_end.astype(int)
    df['is_year_start']= df.date.dt.is_year_start.astype(int)
    #@df['season'] = np.where(df['month'].isin([6,7,8,9]), 1, 0)

    return df


# Setting up variables for input data
@st.cache_data()
def setup(tmp_df_file):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame(
        dict(
            date=[],
            Thermal=[],
            Hydro=[],
            Import=[],
            Export=[],
            Temperature=[],
            Pressure=[],
            Humidity=[],
            Wind_Speed=[],
            timestamp = [],
           
        )
    ).to_csv(tmp_df_file, index=False)

# Setting up a file to save our input data
tmp_df_file = os.path.join(DIRPATH, "tmp", "data.csv")
setup(tmp_df_file)

# setting Title for forms
st.markdown("<h2 style='text-align: center;'> Load Prediction </h2> ", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'> Fill in the details below and click on PREDICT button to make a prediction for a specific date and Demand </h7> ", unsafe_allow_html=True)


# Creating columns for for input data(forms)
left_col, right_col = st.columns(2)

# Developing forms to collect input data
with st.form(key="information", clear_on_submit=True):

    # Setting up input data for 1st column
    left_col.markdown("**FIRST DATA**")

    # Function to calculate Unix timestamp
    def calculate_unix_timestamp(date, time):
        if date and time:
            # Combine date and time to create a datetime object
            selected_datetime = datetime.combine(date, time)

            # Convert the selected datetime to a Unix timestamp
            timestamp = (selected_datetime - datetime(1970, 1, 1)).total_seconds()
            
            return timestamp
        else:
            return None

    # Get the date input from the user
    date = st.date_input('Select a date')

    # Get the time input from the user
    selected_time = st.time_input('Select a time')

    # Calculate Unix timestamp when either date or time changes
    timestamp = calculate_unix_timestamp(date, selected_time)

    # Display the selected datetime in the desired format
    if date and selected_time:
        formatted_datetime = datetime.combine(date, selected_time).strftime('%Y-%m-%d %H:%M:%S')
        st.write('Selected Datetime:', formatted_datetime)

    # Display the corresponding Unix timestamp
    if timestamp is not None:
        st.write('Unix Timestamp:', timestamp)



    Thermal= left_col.number_input("Thermal Power Generated(MW):",min_value=0, max_value=10000)
    Hydro = left_col.number_input("Hydro Power Generated(MW):", min_value=0, max_value=10000)
    Temperature = left_col.number_input("Weather Temperature(°C):", min_value=0, max_value=10000)
    Pressure= left_col.number_input("Weather Pressure(hPa):", min_value=0, max_value=10000)
    
    # Setting up input data for 2nd column
    right_col.markdown("**SECOND DATA**")
    Import = right_col.number_input("Power Imported(MW):", min_value=0, max_value=10000)
    Export = right_col.number_input("Power exported(MW):", min_value=0, max_value=10000)
    Humidity = right_col.number_input("Weather Humidity(%):", min_value=0, max_value=10000)
    Wind_Speed = right_col.number_input("Weather Wind Speed(km/h):", min_value=0, max_value=10000)
 
    submitted = st.form_submit_button("Predict")

# Setting up background operations after submitting forms
if submitted:
    # Saving input data as csv after submissio
    new_data = pd.DataFrame(
        dict(
                date=date,
                Thermal=Thermal,
                Hydro=Hydro,
                Import=Import,
                Export=Export,
                Temperature=Temperature,
                Pressure=Pressure,
                Humidity=Humidity,
                Wind_Speed=Wind_Speed,
                timestamp=timestamp,
            ),
            index=[0],  # Create a DataFrame with a single row
        )
        
     

    existing_data = pd.read_csv(tmp_df_file)
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    updated_data.to_csv(tmp_df_file, index=False)
    st.balloons()

    df = updated_data.copy()
   
        
     # Getting date Features
    processed_data= getDateFeatures(df, 'date')
    processed_data= processed_data.drop(columns=['date'])
    print(processed_data.columns)

    object_cols = processed_data.select_dtypes(['object']).columns
    for col in object_cols:
        processed_data[col] = processed_data[col].astype(float)
   
        
     #Making predictions
    prediction = model.predict(processed_data)
    df['Valco_Demand']= prediction 

    
    # Displaying prediction results
    st.markdown('''---''')
    st.markdown("<h4 style='text-align: center;'> Prediction Results </h4> ", unsafe_allow_html=True)
    st.success(f"Predicted Valco Demand: {prediction[-1]}")
    st.markdown('''---''')

    # Making expander to view all records
    expander = st.expander("See all records") 
    with expander:
        df = pd.read_csv(tmp_df_file)
        df['Valco_Demand']= prediction
        st.dataframe(df)

   # Visualization of prediction trend
    df['datetime'] = df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x) if not np.isnan(x) else np.nan)

    # Now plot the data
    fig = px.line(df, x='datetime', y='Valco_Demand', title='Predicted Valco Demand Over Time')

    # Update x-axis to show date and time
    fig.update_xaxes(
        dtick="M1",
        tickformat="%Y-%m-%d %H:%M:%S",
        tickangle=-45,
        title_text='Date and Time'
    )

    st.plotly_chart(fig)