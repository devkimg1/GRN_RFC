from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, ShuffleSplit, KFold
from ttictoc import tic,toc

all_importances = []
# Pandas for data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pprint import pprint

# Read data as pandas dataframe and display first 5 rows
features = pd.read_csv('full_original_data.csv')
feature_list = list(features.columns)
df_subset = pd.DataFrame(features,columns=list(features.columns))

#print(features.info(null_counts=True))
features.head(5)

print('The shape of our features is:', features.shape)


###CHANGE THESE INDICES
for i in range(1,4512):
# Labels are the values we want to predict (target property)
    tic()
    # Read in data as pandas dataframe and display first 5 rows
    features = pd.read_csv('transcription_factors.csv')
#sid = np.array(features['mp-id'])
    feature_list = list(features.columns)
    df_subset = pd.DataFrame(features,columns=list(features.columns))
    title = "G" + str(i)
    print(title)
    labels = np.array(features[title])

# Remove the target property from the features
# axis 1 refers to the columns
    features= features.drop(title, axis = 1)
    df_subset = df_subset.drop(title, axis = 1)

    features = df_subset.astype(float)
    features = np.array(df_subset)
#print(correlation_matrix)

# Using Skicit-learn to split data into training and testing sets
# Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.15,
                                                                           random_state = 42)



    rf = RandomForestRegressor() #random_state = 42
    

# Look at parameters used by our current forest
    pprint(rf.get_params())


    rf.fit(train_features, train_labels);
    predictions = rf.predict(test_features)
# Print out the mean absolute error (mae)
 

    print(toc())

# Get numerical feature importances
    importances = list(rf.feature_importances_)
    importances.insert(i-1,0)
    all_importances.append(importances)


a = np.asarray(all_importances)
print(all_importances)
###CHANGE FILE NAME TO MATCH INDICES
np.savetxt("importances_tf.csv",a,delimiter=',')