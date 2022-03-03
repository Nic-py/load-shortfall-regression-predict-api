"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed', 'Bilbao_rain_1h', 'Valencia_wind_speed',
       'Seville_humidity', 'Madrid_humidity', 'Bilbao_clouds_all',
       'Bilbao_wind_speed', 'Seville_clouds_all', 'Bilbao_wind_deg',
       'Barcelona_wind_speed', 'Barcelona_wind_deg', 'Madrid_clouds_all',
       'Seville_wind_speed', 'Barcelona_rain_1h', 'Seville_rain_1h',
       'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h',
       'Madrid_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h',
       'Madrid_weather_id', 'Barcelona_weather_id', 'Bilbao_pressure',
       'Seville_weather_id', 'Valencia_pressure', 'Seville_temp_max',
       'Madrid_pressure', 'Valencia_temp_max', 'Valencia_temp',
       'Bilbao_weather_id', 'Seville_temp', 'Valencia_humidity',
       'Valencia_temp_min', 'Barcelona_temp_max', 'Madrid_temp_max',
       'Barcelona_temp', 'Bilbao_temp_min', 'Bilbao_temp',
       'Barcelona_temp_min', 'Bilbao_temp_max', 'Seville_temp_min',
       'Madrid_temp', 'Madrid_temp_min', 'Valencia_wind_deg_level_10',
       'Valencia_wind_deg_level_2', 'Valencia_wind_deg_level_3',
       'Valencia_wind_deg_level_4', 'Valencia_wind_deg_level_5',
       'Valencia_wind_deg_level_6', 'Valencia_wind_deg_level_7',
       'Valencia_wind_deg_level_8', 'Valencia_wind_deg_level_9',
       'Seville_pressure_sp10', 'Seville_pressure_sp11',
       'Seville_pressure_sp12', 'Seville_pressure_sp13',
       'Seville_pressure_sp14', 'Seville_pressure_sp15',
       'Seville_pressure_sp16', 'Seville_pressure_sp17',
       'Seville_pressure_sp18', 'Seville_pressure_sp19',
       'Seville_pressure_sp2', 'Seville_pressure_sp20',
       'Seville_pressure_sp21', 'Seville_pressure_sp22',
       'Seville_pressure_sp23', 'Seville_pressure_sp24',
       'Seville_pressure_sp25', 'Seville_pressure_sp3', 'Seville_pressure_sp4',
       'Seville_pressure_sp5', 'Seville_pressure_sp6', 'Seville_pressure_sp7',
       'Seville_pressure_sp8', 'Seville_pressure_sp9', 'Year', 'Month', 'Day',
       'Day_of_week', 'Start_hour']]

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))