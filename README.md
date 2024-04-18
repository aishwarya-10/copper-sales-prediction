# Copper Sales Prediction:A Machine Learning Approach to Status and Price
This project develops machine learning models that can predict selling prices and classify the status of sales.

<br>

# Copper Sale
The model relies on historical transaction data for copper sales. The data includes details like
1. Item date
2. Delivery date
3. Customer ID
4. Country
5. Status of sale
6. Quantity in tons
7. Item Type, application
8. Dimensions (width, thickness)
9. Reference
10. Selling price per unit.

<br>

# Approach
## Import Necessary Libraries
The libraries needed for copper modeling are entered below.
``` python
#[Data Transformation]
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

#[Data Visualization]
import matplotlib.pyplot as plt
import seaborn as sns

#[Pre-processing]
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # tune up model

#[Balance data]
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#[Model]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score     
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV

#[Metrics]
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score

#[Algorithm]
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier

#[Functions]
import pickle
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```
<br>

## Data Preparation
- EDA analysis shows the data requires scaling, filling missing values, outliers removal, delivery date correctness, and data type conversion.
- The columns (object datatype) that cannot be filled and have more than 50% missing value are dropped.
- The datatypes of certain columns (objects) are converted to numeric and datetime format.
- The missing values of categorical features are filled using `mode` and continuous features are filled using the median.
- The categorical variables are converted to numerical format using `ordinal encoding`.
- The skewed data [quantity, thickness, selling price] are visualized and transformed to a normal distribution using the `log transform` method.
- The outliers in the data are removed by the Interquartile Range (IQR) method.
- The correlation analysis shows the data has less correlation and hence no column is dropped.
- The wrong delivery date in the dataset is handled by the regression method.
- Finally, the processed data is saved for further analysis.

<br>

## Predict Status
- 'status' is categorical data and hence classification technique is used for predicting "won" or "lost".
- The data was imbalanced and used `oversampling` method to balance the data.
- The best classification model is evaluated using ROCAUC score, accuracy, and F1 score.
- `ExtraTreesClassifier` has produced high accuracy than all other tree regression models with 
20% test size.
- The model is saved using `pickle` library.

<br>

## Predict Selling Price
- 'selling_price' is a continuous data and hence regression technique is used.
- ExtraTreesRegressor` performed well with 0.95 R^2 value and low error.
- The trained model is saved using pickle for further predictions.

<br>

## Dashboard
Streamlit dashboard which inputs all the features required for the model to predict is made for status and selling price prediction.
