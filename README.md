# DATA-PREPROCESSING
# AIM
To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset, and applying feature scaling. 
# Procedure
1.	Import the required Python libraries. 
2.	Mount Google Drive and load the dataset using Pandas. 
3.	Display the first few records of the dataset. 
4.	Inspect the dataset using df.info() and df.shape. 
5.	Separate the independent variables (X) and dependent variable (Y). 
6.	Convert the independent variables into an array. 
7.	Identify and handle missing values using SimpleImputer with the mean strategy. 
8.	Encode the categorical Country column using LabelEncoder. 
9.	Apply One-Hot Encoding to convert categorical country values into dummy variables. 
10.	Encode the dependent variable Purchased using LabelEncoder. 
11.	Split the dataset into training and testing sets using train_test_split. 
12.	Apply Standard Scaler for feature scaling. 
13.	Display the preprocessed training and testing datasets. 

# Program
import pandas as pd 

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import LabelEncoder 

from sklearn.compose import ColumnTransformer 

from sklearn.preprocessing import OneHotEncoder 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

#Load dataset 

dataset = pd.read_csv("Data.csv") 

#Display first five rows 

print("First five rows:") 

print(dataset.head()) 

#Display dataset information 

print("\nDataset Information:") 

dataset.info() 

#Display shape 

print("\nDataset Shape:") 

print(dataset.shape) 

#Separate independent and dependent variables 

X = dataset.iloc[:, :-1].values 

y = dataset.iloc[:, -1].values 

#Handle missing values 

imputer = SimpleImputer(strategy="mean") 

X[:, 1:3] = imputer.fit_transform(X[:, 1:3]) 

#Encode categorical independent variable 

label_encoder = LabelEncoder() X[:, 0] = label_encoder.fit_transform(X[:, 0]) 

#Apply One-Hot Encoding 

ct = ColumnTransformer( transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough" ) X = ct.fit_transform(X) 

#Encode dependent variable 

label_encoder_y = LabelEncoder() 

y = label_encoder_y.fit_transform(y) 

#Split dataset 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1 ) 

#Feature scaling 

scaler = StandardScaler() 

X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 

#Display results 

print("\nPreprocessed Training Data:") 

print(X_train) 

print("\nPreprocessed Testing Data:") 

print(X_test) 

print("\nTraining Target Values:") 

print(y_train) 

print("\nTesting Target Values:") 

print(y_test)

# EXPECTED OUTPUT

The program displays:

1.First few records of the dataset.

2.Dataset information and data types.

3.Number of rows and columns.

4.Preprocessed training dataset.

5.Preprocessed testing dataset.

6.Encoded target values.

After preprocessing, the dataset contains numerical and scaled values that are suitable for use in machine learning algorithms.

# RESULT
The given dataset was successfully preprocessed using Python and Scikit-learn. Missing values were handled using SimpleImputer, categorical data was converted into numerical form using Label Encoding and One-Hot Encoding, and the dataset was divided into training and testing sets.

Feature scaling was then performed using StandardScaler. The resulting preprocessed dataset is suitable for further machine learning model development and analysis.
