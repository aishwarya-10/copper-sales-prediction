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
## Step 1: Import Necessary Libraries
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

## Step 2: Data Preparation
Exploratory Data Analysis (EDA) revealed the data required several cleaning steps before further analysis. These steps included:
**1. Missing Value Handling:**
Columns with an object data type and more than 50% missing values that couldn't be effectively filled were dropped.
Missing values in categorical features were imputed using the most frequent value (`mode`).
Missing values in continuous features were filled using the `median` value.

**2. Data Type Conversion:**
Data types of specific object columns were converted to numeric formats or datetime formats for better analysis.

**3. Feature Engineering:**
Categorical variables were transformed into numerical representations using `ordinal encoding`. This allows for calculations and modeling techniques that work with numerical data.

**4. Normalization:**
Skewed data in features like quantity, thickness, and selling price were identified through visualization.
A `log transformation` was applied to these features to achieve a more normal distribution, which is often preferred for statistical modeling.

**5. Outlier Treatment:**
Outliers were identified and removed using the `Interquartile Range` (IQR) method. This helps maintain data integrity and prevents outliers from unduly influencing model results.

**6. Correlation Analysis:**
Correlation analysis revealed a low level of correlation between features, indicating no significant redundancy. Therefore, no feature removal was necessary at this stage.

**7. Delivery Date Correction:**
Inaccurate delivery dates within the dataset were addressed using a regression method. This method can help predict and correct the most likely values for these discrepancies.

**8. Data Persistence:**
Finally, the cleaned and processed data was saved for further analysis and model training.

<br>

## Step 3: Predict Status
- The "status" variable is categorical, indicating a win or loss, a classification technique was chosen for prediction.
- The data exhibited an imbalance between the number of "won" and "lost" cases. To address this, the `oversampling` method was employed to create a more balanced dataset. This ensures the model doesn't get biased towards the more frequent class.
- To identify the best-performing classification model, we evaluated them using a combination of metrics:
  - **ROCAUC score:** Measures the model's ability to distinguish between "won" and "lost" cases.
  - **Accuracy:** The overall percentage of correct predictions.
  - **F1 score:** A balanced metric considering both precision and recall, especially important for imbalanced datasets.
- Among the tree-based regression models tested with a 20% test set size, the `ExtraTreesClassifier` achieved the highest accuracy. This suggests it generalizes well to unseen data and avoids overfitting.
- The final chosen model was saved using the `pickle` library for future use and potential deployment.

<br>

## Step 4: Predict Selling Price
- The 'selling_price' variable is continuous data and hence regression technique was chosen for prediction.
- `ExtraTreesRegressor` performed well with 0.95 R^2 value and low error.
- The trained model is saved using pickle for further predictions.

<br>

## Step 5: Dashboard
Streamlit dashboard which inputs all the features required for the model to predict is made for status and selling price prediction.
