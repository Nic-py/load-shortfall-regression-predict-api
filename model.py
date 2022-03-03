"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
from mailbox import Message
from re import X
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, r2_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    df_list = list()
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    # for i in range(0,len(feature_vector_dict)):
    # df_list.append( pd.DataFrame.from_dict([feature_vector_dict[i]]))
    # feature_vector_df1 = pd.concat(df_list ,ignore_index=True)
    #print(type(feature_vector_df))
    #feature_vector_df = data
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    #load the data

    df = feature_vector_df.copy()
    features = ['Madrid_wind_speed', 'Bilbao_rain_1h',
       'Valencia_wind_speed', 'Seville_humidity', 'Madrid_humidity',
       'Bilbao_clouds_all', 'Bilbao_wind_speed', 'Seville_clouds_all',
       'Bilbao_wind_deg', 'Barcelona_wind_speed', 'Barcelona_wind_deg',
       'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_rain_1h', 'Bilbao_snow_3h',
       'Barcelona_pressure', 'Seville_rain_3h', 'Madrid_rain_1h',
       'Barcelona_rain_3h', 'Valencia_snow_3h', 'Madrid_weather_id',
       'Barcelona_weather_id', 'Bilbao_pressure', 'Seville_weather_id',
        'Seville_temp_max', 'Madrid_pressure',
       'Valencia_temp_max', 'Valencia_temp', 'Bilbao_weather_id',
       'Seville_temp', 'Valencia_humidity', 'Valencia_temp_min',
       'Barcelona_temp_max', 'Madrid_temp_max', 'Barcelona_temp',
       'Bilbao_temp_min', 'Bilbao_temp', 'Barcelona_temp_min',
       'Bilbao_temp_max', 'Seville_temp_min', 'Madrid_temp', 'Madrid_temp_min']
    # 'Valencia_wind_deg_level_10',
    #    'Valencia_wind_deg_level_2', 'Valencia_wind_deg_level_3',
    #    'Valencia_wind_deg_level_4', 'Valencia_wind_deg_level_5',
    #    'Valencia_wind_deg_level_6', 'Valencia_wind_deg_level_7',
    #    'Valencia_wind_deg_level_8', 'Valencia_wind_deg_level_9',
    #    'Seville_pressure_sp10', 'Seville_pressure_sp11',
    #    'Seville_pressure_sp12', 'Seville_pressure_sp13',
    #    'Seville_pressure_sp14', 'Seville_pressure_sp15',
    #    'Seville_pressure_sp16', 'Seville_pressure_sp17',
    #    'Seville_pressure_sp18', 'Seville_pressure_sp19',
    #    'Seville_pressure_sp2', 'Seville_pressure_sp20',
    #    'Seville_pressure_sp21', 'Seville_pressure_sp22',
    #    'Seville_pressure_sp23', 'Seville_pressure_sp24',
    #    'Seville_pressure_sp25', 'Seville_pressure_sp3', 'Seville_pressure_sp4',
    #    'Seville_pressure_sp5', 'Seville_pressure_sp6', 'Seville_pressure_sp7',
    #    'Seville_pressure_sp8', 'Seville_pressure_sp9', 'Year', 'Month', 'Day',
    #    'Day_of_week', 'Start_hour']

    for col in ['Valencia_wind_deg', 'Seville_pressure']:
        df[col] = df[col].astype('category')

#have a copy of our train_data with the correct  dtype for our categorical columns
    df_copy = df[features].copy()
    #df_copy['time'] = pd.to_datetime(df_copy['time'])
    #df_dummies = pd.get_dummies(df_copy, drop_first = True)
    
    # create new features
    # year
    #df_dummies['time'] = pd.to_datetime(df_dummies['time'])
    # df_dummies['Year'] = df_dummies['time'].dt.year
    # # month
    # df_dummies['Month'] = df_dummies['time'].dt.month
    # # day
    # df_dummies['Day'] = df_dummies['time'].dt.day
    # # Dayofweek
    # df_dummies['Day_of_week'] = df_dummies['time'].dt.weekday
    # # hour
    # df_dummies['Start_hour'] = df_dummies['time'].dt.hour
    # # Drop Feature
    # df_dummies = df_dummies.drop(['time','Unnamed: 0'] , axis=1)
    # print(df_dummies.shape)


    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(df_copy.values)
    IterativeImputer(random_state=0)
    

    imputed = imp.transform(df_copy.values)

    predict_vector = pd.DataFrame(imputed, index=df_copy.index, columns=df_copy.columns)
    print(predict_vector.shape)
    
        
    # ------------------------------------------------------------------------
    
    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
#_preprocess_data(pd.read_csv('utils/data/df_train.csv'))