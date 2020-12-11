from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, ShuffleSplit, KFold
from ttictoc import tic,toc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

all_importances = []

# Read in data as pandas dataframe and display first 5 rows
features = pd.read_csv('full_original_data.csv')

feature_list = list(features.columns)

df_subset = pd.DataFrame(features,columns=list(features.columns))
features.head(5)

print('The shape of our features is:', features.shape)

###CHANGE THESE INDICES
for i in range(1,4512):
# Labels are the values we want to predict (target property)
    tic()
    # Read in data as pandas dataframe and display first 5 rows
    features = pd.read_csv('full_original_data.csv')
    features_a = np.asarray(features)
    features_tf = features_a[:,0:334]
    features_tf = pd.DataFrame(features_tf)

    feature_list = list(features_tf.columns)

    df_subset = pd.DataFrame(features,columns=list(features.columns))
    title = "G" + str(i)
    print(title)
    labels = np.array(features[title])
    train_labels = pd.DataFrame(labels)


    rf = RandomForestRegressor() #random_state = 42
    

# Look at parameters used by our current forest
    pprint(rf.get_params())
    rf.fit(features_tf, train_labels.ravel());

# Print out the mean absolute error (mae)
 

    print(toc())

# Get numerical feature importances
    importances = list(rf.feature_importances_)
    all_importances.append(importances)


a = np.asarray(all_importances)

###CHANGE FILE NAME TO MATCH INDICES
np.savetxt("importances_tf_allgenes.csv",a,delimiter=',')