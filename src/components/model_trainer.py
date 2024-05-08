import os
import sys
from dataclasses import dataclass
from src.components.model_hyperparameters import (
    catboost_hyperparams,
)


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    random_seed = 42

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X ,y):
        try:
            models = {
                "Random Forest": RandomForestRegressor(random_state=self.model_trainer_config.random_seed),
                "Decision Tree": DecisionTreeRegressor(random_state=self.model_trainer_config.random_seed),
                "Gradient Boosting": GradientBoostingRegressor(random_state=self.model_trainer_config.random_seed),
                "Linear Regression": LinearRegression(),
                #"XGBRegressor": XGBRegressor(), #xgboost library not tested
                "LightGBM Regressor": LGBMRegressor(random_state=self.model_trainer_config.random_seed, verbosity=-1),
                "CatBoosting Regressor": CatBoostRegressor(random_state=self.model_trainer_config.random_seed, verbose=False, **catboost_hyperparams),
            }

            # Evaluate models
            model_report = evaluate_models(X, y, models)

            # Find best model
            best_model_name = min(model_report, key=lambda k: model_report[k]['mean_rmse'])
            best_model = models[best_model_name]
            best_rmse = model_report[best_model_name]['mean_rmse']

            logging.info(f"Best model found: {best_model_name}, RMSE: {best_rmse}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
            # Save all models and their RMSE scores
            for model_name, report in model_report.items():
                model = models[model_name]
                rmse = report['mean_rmse']
                logging.info(f"{model_name}: RMSE = {rmse}")

                # Save the model
                save_object(
                    file_path=f"{self.model_trainer_config.trained_model_file_path}_{model_name}.pkl",
                    obj=model
                )
            print(best_model, best_rmse)            
            return model_report, best_model, best_rmse
            

        except Exception as e:
            raise CustomException(e, sys)