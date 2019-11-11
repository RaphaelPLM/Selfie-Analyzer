# Pandas is used for data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot

# Read the .txt with selfie data
data = pd.read_csv('data/selfie_dataset.txt', sep=" ", header=None)

# Specifies the name of each column
data.columns = ["image_name",
            "popularity_score",
            "partial_faces",
            "is_female",
            "baby",
            "child",
            "teenager",
            "youth",
            "middle_age",
            "senior",
            "white",
            "black",
            "asian",
            "oval_face",
            "round_face",
            "heart_face",
            "smiling",
            "mouth_open",
            "frowning",
            "wearing_glasses",
            "wearing_sunglasses",
            "wearing_lipstick",
            "tongue_out",
            "duck_face",
            "black_hair",
            "blond_hair",
            "brown_hair",
            "red_hair",
            "curly_hair",
            "straight_hair",
            "braid_hair",
            "showing_cellphone",
            "using_earphone",
            "using_mirror",
            "braces",
            "wearing_hat",
            "harsh_lighting",
            "dim_lighting"]

# Popularity is what we are trying to predict
labels = np.array(data['popularity_score'])

data = data.drop('popularity_score', axis = 1)
data = data.drop('image_name', axis = 1)

# Before converting the data table to an numpy array, store the names of the columns
attributes_list = list(data.columns)

# Convert data to an numpy.array
data = np.array(data)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.3, random_state = 42)

# Specifies the algorithm used in prediction. 
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(train_data, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_data)

# Calculate the absolute errors
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'points.')

mape = 100 * (errors / test_labels)# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 4), '%.')

# Gets the relevance of each attribute. This allow us to evaluate what makes a selfie popular.
relevance_list = list(rf.feature_importances_)

attribute_relevance = [(data, round(attribute, 5)) for data, attribute in zip(attributes_list, relevance_list)]# Sort the feature relevance_list by most important first
attribute_relevance = sorted(attribute_relevance, key = lambda x: x[1], reverse = True)# Print out the feature and relevance_list 


[print('Variable: {:23} Importance: {}'.format(*pair)) for pair in attribute_relevance];