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
│   ├── 
│   ├── 
│   ├── 
│   ├── 
│   └── 
│
├── artifacts/
│   └── 
│
└── README.md
```

## Instructions for executing the pipeline and modifying any parameters
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Ensure Python 3.x is installed.
4. Install the required dependencies by running `pip install -r requirements.txt`
5. Open a command prompt terminal and activate environement in the folder with conda activate venv/
6. type in the terminal `python src/components/main.py`
7. check the logs folder if there are exceptions otherwise it should show (insert image)
8. type in the terminal `python app.py`
9. This will deploy a simple webapp (connect to URL 127.0.0.1:5000/predictvalue)
10. Select the available options and click submit to obtain a prediction from the model


## Description of pipeline workflow
The pipeline consists of the following steps:

1. `Data_Ingestion.py`
- 


2. `Data_transformation.py`
-

3. `Model_trainer.py`
- Train multiple regression models using various algorithm.
- Evaluate each model using cross-validation.
- Select the best-performing model based on evaluation metric RMSE

4. `app.py`
- Use the trained model to predict Football Player's value in KEUR for new data.


## Overview of key findings from Exploratory Data Analysis


## Explaination of choice of models


## Evaluation of models developed
The models are evaluated using the following metric:

- Root Mean Squared Error (RMSE)

> ${RMSE} = \sqrt{\frac{1}{n} sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

