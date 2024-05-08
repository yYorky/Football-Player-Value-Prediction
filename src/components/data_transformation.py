import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,RobustScaler


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

           

    def get_data_transformer_object(self,df):
        '''
        This function is responsible for data transformation pipline
        '''
        try:

            numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")
                        
            
            class DropColumn(BaseEstimator, TransformerMixin):
                def __init__(self, cols=[]):
                    self.cols = cols
                def fit(self, X, y=None):
                    return self
                def transform(self, X, y=None):
                    return X.drop(self.cols, axis=1)
                
                        
            preprocessor= Pipeline([
                ('drop', DropColumn(cols=['Height', 'Weight', 'Growth', 'Total attacking', 'Crossing', 'Heading accuracy', 'Short passing', 'Volleys', 'Total skill', 'Dribbling', 'Curve', 'FK Accuracy',
                              'Long passing', 'Ball control', 'Acceleration', 'Agility', 'Reactions', 'Balance', 'Total power', 'Shot power', 'Jumping', 'Stamina', 'Strength', 'Long shots',
                              'Total mentality', 'Aggression', 'Interceptions', 'Att. Position', 'Vision', 'Penalties', 'Total defending', 'Defensive awareness', 'Standing tackle', 
                              'Sliding tackle', 'Total goalkeeping', 'Total stats', 'Base stats', 'Pace / Diving', 'Shooting / Handling', 'Passing / Kicking', 'Dribbling / Reflexes', 
                              'Defending / Pace',
                              'GK Kicking', 'GK Positioning', 'GK Handling', 'GK Reflexes', 'GK Diving','foot'
                    ])),
                ('prep', ColumnTransformer([
                    ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                     list(set(categorical_features)
                          -set(['foot'
                              
                              ])
                              )),],
                              remainder='passthrough').set_output(transform='pandas')),
                              ('scale',RobustScaler().set_output(transform='pandas'))
                              ])
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

            
    



    def initiate_data_transformation(self,train_path):
        try:
            df=pd.read_csv(train_path)
            
            logging.info("Read train data completed")
            
            # Drop specified columns
            columns_to_drop = ['Unnamed: 64', 'name', 'Team & Contract']
            df = df.drop(columns=columns_to_drop)
            logging.info("Drop ['Unnamed: 64', 'name', 'Team & Contract'] columns")


            df = df.loc[~df.index.duplicated()].copy()
            logging.info("Drop duplicated rows(index)")

            # Define a function to handle values in the format 'number+number'
            def handle_plus_minus(value):
                if isinstance(value, str) and '+' in value:
                    parts = value.split('+')
                    return int(parts[0]) + int(parts[1])
                elif isinstance(value, str) and '-' in value:
                    parts = value.split('-')
                    return int(parts[0]) - int(parts[1])
                else:
                    return value

            # Define the list of variables to convert to numeric and perform operations
            numeric_variables = ['Overall rating', 'Potential', 'Crossing', 'Finishing', 'Heading accuracy', 'Short passing', 
                                'Volleys', 'Dribbling', 'Curve', 'FK Accuracy', 'Long passing', 'Ball control', 
                                'Acceleration', 'Sprint speed', 'Agility', 'Reactions', 'Balance', 'Shot power', 
                                'Jumping', 'Stamina', 'Strength', 'Long shots', 'Aggression', 'Interceptions', 
                                'Att. Position', 'Vision', 'Penalties', 'Composure', 'Defensive awareness', 
                                'Standing tackle', 'Sliding tackle', 'GK Diving', 'GK Handling', 'GK Kicking', 
                                'GK Positioning', 'GK Reflexes']

            # Apply the custom function to handle values in the specified columns
            df[numeric_variables] = df[numeric_variables].applymap(handle_plus_minus)
            df[numeric_variables] = df[numeric_variables].apply(pd.to_numeric, errors='coerce')
            logging.info("Summation of Columns into Numeric")

            # Define a function to extract height in cm
            def extract_height(height):
                if isinstance(height, str):
                    # Split the string by '/'
                    parts = height.split('/')
                    # Extract the part with cm
                    cm_part = parts[0].strip()
                    # Extract the integer value for cm
                    height_cm = int(cm_part.replace('cm', '').strip())
                    return height_cm
                else:
                    return height

            # Define a function to extract weight in kg
            def extract_weight(weight):
                if isinstance(weight, str):
                    # Split the string by '/'
                    parts = weight.split('/')
                    # Extract the part with kg
                    kg_part = parts[0].strip()
                    # Extract the integer value for kg
                    weight_kg = int(kg_part.replace('kg', '').strip())
                    return weight_kg
                else:
                    return weight

            # Apply the custom functions to extract height and weight
            df['Height'] = df['Height'].apply(extract_height)
            df['Weight'] = df['Weight'].apply(extract_weight)

            # Define a function to convert value strings to numeric
            def convert_value(value):
                if isinstance(value, str):
                    if value[-1] == 'M':
                        return float(value[1:-1].replace('M', '')) * 1000
                    elif value[-1] == 'K':
                        return float(value[1:-1].replace('K', '')) 
                    else:
                        return float(value[1:])
                else:
                    return value

            # Define a function to convert wage strings to numeric
            def convert_wage(wage):
                if isinstance(wage, str):
                    if wage[-1] == 'K':
                        return float(wage[1:-1].replace('K', '')) 
                    else:
                        return float(wage[1:])
                else:
                    return wage

            # Define a function to convert release clause strings to numeric
            def convert_release_clause(release_clause):
                if isinstance(release_clause, str):
                    if release_clause[-1] == 'M':
                        return float(release_clause[1:-1].replace('M', '')) * 1000
                    elif release_clause[-1] == 'K':
                        return float(release_clause[1:-1].replace('K', '')) 
                    else:
                        return float(release_clause[1:])
                else:
                    return release_clause

            # Apply the custom functions to convert value, wage, and release clause
            df["Value('000)"] = df['Value'].apply(convert_value)
            df["Wage('000)"] = df['Wage'].apply(convert_wage)
            df["Release clause('000)"] = df['Release clause'].apply(convert_release_clause)


            # Drop the specified columns
            df = df.drop(columns=['Value', 'Wage', 'Release clause'])

            logging.info("Convert height, Weight, Value,Wage and release clause")      

                     

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object(df)

            target_column_name="Value('000)"
            X = df.drop(columns=[target_column_name],axis=1)
            y = df[target_column_name]                   
                       

            logging.info(f"Applying preprocessing object on dataframe.")

            X =preprocessing_obj.fit_transform(X)

            
                       
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj

            )

            return (
                X,y,                
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)