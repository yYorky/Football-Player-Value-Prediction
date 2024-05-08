# Football-Player-Value-Prediction

<p align="center">
  <img src="https://static0.givemesportimages.com/wordpress/wp-content/uploads/2024/01/study-names-the-10-most-expensive-players-in-the-world-image-1.jpg" alt="image"/>
</p>

## Overview of Project
Football clubs, analysts, fantasy enthusiasts, and bettors often face challenges in assessing player values accurately. To address this, we've developed a machine learning tool leveraging data from sofifa.com to predict footballers' market values. This tool serves diverse user needs, including scouting, performance evaluation, and strategic decision-making.

### Folder Structure

```bash
project-root/
│
├── data/
│   ├── players_3120.csv
│   └── players_17937.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── experiment.ipynb
│
├── src/
│   ├── components
│   ├── pipeline
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── static
│
├── templates
│   ├── home.html
│   └── index.html
│
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
└── setup.py
```

## Instructions for executing the pipeline and modifying any parameters
1. Clone the repository to your local machine.
2. Navigate to the project directory in anaconda prompt using  `cd C:\Users\YourName\Desktop\Football Player Value Prediction`
3. Create an environment using `conda create -p venv python==3.8 -y`, an environment called `venv` should be added to your local folder.
4. Activate the environement in the folder with `conda activate venv/`
5. Install the required dependencies by running `pip install -r requirements.txt`
6. To run the model and generate the pkl file needed to deploy the webapp, type in the terminal `python src/components/main.py`.
7. check the logs folder if there are exceptions otherwise it should show something like the following.
<p align="center">
  <img src="https://github.com/yYorky/aiap-yeo-york-yong-163A/blob/main/static/main_py_successful_log.JPG" alt="image"/>
</p>

8. To run the webapp, type in the terminal `python app.py`
9. This will deploy a simple webapp (connect to URL 127.0.0.1:5000/predictvalue)
10. Select the available options and click submit to obtain a prediction from the model.
<p align="center">
  <img src="https://github.com/yYorky/aiap-yeo-york-yong-163A/blob/main/static/main_py_successful_log.JPG" alt="image"/>
</p>


## Description of pipeline workflow
The pipeline consists of the following steps:

1. `data_Ingestion.py`
- Logs the entry into the data ingestion component.
- Reads data from a CSV file into a DataFrame (df).
- Logs the successful reading of the dataset.
- Creates directories if they don't exist for the target file path.
- Saves the DataFrame to the specified path.
- Logs the completion of data ingestion.
- Returns the path where the data is saved.
- Handles exceptions by raising a CustomException and logging errors.

2. `data_transformation.py`
- Reads the dataset from a given path.
- Drops specified columns and removes duplicate rows.
- Defines custom functions to handle specific column values and applies them.
- Converts string values to numeric for specific columns.
- Constructs preprocessing objects for feature transformation.
- Saves the preprocessing object to a file.
- Returns transformed features (X), target variable (y), and the path to the saved preprocessing object.

3. `model_trainer.py`
- Initializes a dictionary of regression models with their corresponding names.
- Trains each model using the given features (X) and target variable (y).
- Evaluates the performance of each model using RMSE (Root Mean Squared Error).
- Identifies the best-performing model based on RMSE.
- Saves the best model and all trained models along with their RMSE scores.
- Returns a report of model performance, the best model, and its RMSE score.

4. `model_hyperparameter.py`
- These hyperparameters are obtained from experimentation and are used to configure the CatBoostRegressor during model training.

5. `main.py`
-  This script orchestrates the data ingestion, data transformation, and model training stages.


## Overview of key findings from Exploratory Data Analysis

### Insights for Data Preprocessing
| No. | Action | Reason|
| --- | ------ | ------|
|1| Drop Columns: Unnamed: 64, Name, Team & Contract| Nan Values, or not useful|
|2| Drop duplicated rows | Duplicated rows needds to be removed|
|3| For these variable we need to make the summation or subtraction for some entries and convert to numeric: Overall rating, potential, Crossing, Finishing, Heading accuracy, Short passing, Volleys, Dribbling, Curve, FK Accuracy, Long passing, Ball Control, Acceleration, Sprint speed, Agility, Reactions, Balance, Shot power, Jumping, Stamina, Strength, Long Shots, Aggression, Interceptions, Att. Position, Vision, Penalties, Composure, Defensive awareness, Standing tackle, Sliding tackle', GK Diving, GK Handling, GK Kicking, GK Positioning, GK Reflexes | Convert object to numeric|
|4|  For these features we need to treat: Height (take the interger for cm) , Weight (take the interger for kg), Value (convert from to numeric), Wage (convert to numeric), Release Clause (convert to numeric),| Convert object to numeric |

### Insights for Feature Selection and other consideration

| No. | Action | Reason|
| --- | ------ | ------|
|1| Perform feature selection based on feature importance, drop columns with high VIF | Many columns with high VIF and multicollinerity | 
|2| Outliers in data | Perform Robust Scaling|

## Explaination of choice of models
| Model                    | Strengths                                                                                   | Applicability                                                                                                  | RMSE Consideration                                                                                                     |
|--------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Random Forest Regressor  | - Handles high-dimensional data well<br>- Robust to overfitting<br>- Ignores irrelevant features | - Suitable for complex, nonlinear relationships<br>- Requires minimal data preprocessing<br>- Handles both numerical and categorical features | Tends to provide good performance and often yields low RMSE values                                                    |
| Decision Tree Regressor  | - Easy to interpret and understand<br>- Handles both numerical and categorical data<br>- Requires minimal data preparation | - Suitable for capturing complex relationships<br>- Handles interactions between features well | Can provide competitive performance for regression tasks, although may overfit the training data                      |
| Gradient Boosting Regressor | - Combines predictions of multiple weak learners<br>- Robust to outliers<br>- Can handle nonlinear relationships | - Suitable for regression tasks requiring high accuracy<br>- Captures complex patterns in data<br>- Performs well in practice | Often yields low RMSE values and can provide state-of-the-art performance for regression tasks                       |
| Linear Regression        | - Simple, interpretable, and computationally efficient<br>- Well-suited for problems with linear relationships between features and target variable | - Suitable for tasks with approximately linear relationships between features and target variable | Provides a baseline performance for comparison with more complex models; limited performance if data exhibits nonlinear relationships |
| LightGBM Regressor       | - Efficient, fast, and scalable<br>- Handles large datasets and high-dimensional features | - Suitable for regression tasks requiring high performance and efficiency<br>- Useful for large datasets | Often yields low RMSE values and provides competitive performance compared to other gradient boosting algorithms         |
| CatBoost Regressor       | - Robust to categorical features and missing data<br>- Performs well without extensive data preprocessing<br>- Robust to overfitting | - Suitable for regression tasks with categorical features and missing values<br>- Performs well with minimal data preprocessing | Often yields low RMSE values and provides excellent performance, especially when dealing with categorical features       |




## Evaluation metrics for consideration
For the context of this problem, there are several metrics that can be considered.

1. Mean Absolute Error (MAE): This metric measures the average absolute difference between predicted and actual scores. It provides a straightforward interpretation of the model's prediction errors in the same unit as the target variable (examination scores).
2. Root Mean Squared Error (RMSE): RMSE is similar to MAE but gives higher weight to large errors. It penalizes larger errors more heavily, making it suitable for cases where large errors are particularly undesirable.
3. Mean Squared Error (MSE): MSE is the average of the squared differences between predicted and actual scores. It provides a measure of the average squared error of the model's predictions and is useful for understanding the spread of prediction errors.
4. R-squared (R2) Score: R2 score represents the proportion of the variance in the target variable that is explained by the model. It indicates how well the independent variables explain the variability of the dependent variable. A higher R2 score indicates a better fit of the model to the data.

## Evaluation metrics selected for this modelling
RMSE (Root Mean Squared Error) is selected as the suitable metric for predicting football player prices due to several reasons. 

${RMSE} = \sqrt{\frac{1}{n} sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$


1. First, in the context of player prices, where the goal is to accurately estimate the monetary value of a player, RMSE provides a direct measure of the prediction error in the same units as the target variable (i.e., currency). This makes RMSE intuitive and easy to interpret, as it represents the average magnitude of the errors in predicting player prices.

2. Second, RMSE considers both the magnitude and direction of errors, making it a comprehensive measure of model performance. By squaring the errors before taking the square root, RMSE penalizes larger errors more heavily than smaller ones. In the context of football player prices, where accurately predicting high-value players may be more critical, RMSE effectively captures the impact of these larger errors on overall model performance.

3. Additionally, RMSE is widely used and accepted in regression tasks, making it suitable for comparing different models and assessing their predictive accuracy. Its popularity stems from its simplicity and robustness, as it provides a single, easy-to-understand metric that summarizes the overall performance of a regression model.




## Acknowledgements

- Yulas Ozen’s Medium Article on Predicting Football Players’ Market Value Using Machine Learning: https://yulasozen.medium.com/predicting-football-players-market-value-using-machine-learning-b28be298e91e#:~:text=The%20application%20of%20linear%20regression,and%20contribute%20to%20accurate%20forecasts.
- Analytics Vidhya’s Article on Machine Learning Project for Predicting Football Players’ Market Value: https://medium.com/analytics-vidhya/machine-learning-project-predicting-football-players-market-value-fd40636462bf
- Tola Adelase’s GitHub Repository on Predicting Market Value of Football Players Using Machine Learning Algorithm: https://github.com/Tola-adelase/predicting-market-value-of-football-players-using-machine-learning-algorithm
- Football Benchmark’s Library on Player Valuation and Transfer Market Analysis: https://www.footballbenchmark.com/library/player_valuation_putting_data_to_work_on_transfer_market_analysis
- Research Paper: “Predicting Football Players’ Market Value” : https://arxiv.org/abs/2206.13246
- GitHub Repository by Ohad Mavdali on Predicting Football Players’ Value Using ML: https://github.com/ohadmavdali/Predict-football-players-value-using-ML
- Research Paper: “Machine Learning for Football Player Valuation”: https://arxiv.org/abs/2403.07669
- Yuan He’s Undergraduate Research on Football Player Valuation: https://www.stat.berkeley.edu/~aldous/Research/Ugrad/Yuan_He.pdf
- Thesis by Arno van den Ven on Football Player Valuation: https://arno.uvt.nl/show.cgi?fid=161188

