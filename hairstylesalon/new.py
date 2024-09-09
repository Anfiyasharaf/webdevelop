## Importing necessary libraries
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computations and arrays
import matplotlib.pyplot as plt  # Creating visualizations
import seaborn as sns  # Advanced data visualization library
import plotly.express as px  # Interactive visualizations
import missingno as msno  # Visualizing missing data patterns

from sklearn.pipeline import make_pipeline  # Creating ML pipelines
from sklearn.preprocessing import StandardScaler  # Standardizing features
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
from sklearn.linear_model import LogisticRegression  # Logistic regression classifier
from sklearn.neighbors import KNeighborsClassifier  # K-nearest neighbors classifier
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.tree import DecisionTreeClassifier  # Decision tree classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier  # Ensemble classifiers
from sklearn.naive_bayes import GaussianNB  # Naive Bayes classifier
from sklearn.neural_network import MLPClassifier  # Multilayer Perceptron classifier
from xgboost import XGBClassifier  # XGBoost classifier
from lightgbm import LGBMClassifier  # LightGBM classifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, classification_report  # ML metrics

import joblib  # Saving and loading ML models
import warnings  # Controlling warning behavior

warnings.filterwarnings('ignore')  # Ignoring warnings

# Additional imports
from sklearn import preprocessing  # Data preprocessing
from sklearn import metrics  # Evaluation metrics
import random

np.random.seed(42)
random.seed(42)

# Setting up the plot style and options
plt.style.use('fivethirtyeight')

# Example of using confusion matrix
# Assuming `y_true` and `y_pred` are your true and predicted labels
y_true = [0, 1, 0, 1, 1, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]

# Compute confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plot_confusion_matrix(conf_mat, cmap=plt.cm.Blues)
plt.show()
