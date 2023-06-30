import os
import numpy as np
import pandas as pd

key = 'hog'
mf = '100X'
folder_path = f'./features/all/{mf}/stat/{key}/'

# Count the number of csv files in the folder
number_of_images = sum(1 for file_name in os.listdir(folder_path) if file_name.endswith('.csv'))

# Assuming all csv files have the same number of features (columns)
number_of_features = 1

csv_file_path = os.path.join(folder_path, os.listdir(folder_path)[0])
number_of_rows = pd.read_csv(csv_file_path).shape[0]
number_of_features += number_of_rows

feature_matrix = np.zeros((number_of_images, number_of_features))
image_names = []

for i, file_name in enumerate(os.listdir(folder_path)):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        image_name = file_name[1:-4]  # Remove the last 4 characters (.csv) from the file name
        image_names.append(image_name)
        features = np.loadtxt(file_path, delimiter=',')
        feature_matrix[i, :] = features

target_var = feature_matrix[:, -1]
feature_matrix = feature_matrix[:, :-1] 

# Perform feature selection (example using mutual information)
from sklearn.feature_selection import SelectKBest, f_regression

# Set the desired number of features to keep
num_features_to_keep = 1000

# Perform feature selection using SelectKBest and f_regression
selector = SelectKBest(score_func=f_regression, k=num_features_to_keep)
selected_features = selector.fit_transform(feature_matrix, target_var)

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)

# Filter the feature_matrix to keep only the selected features
selected_feature_matrix = feature_matrix[:, selected_indices]

print(selected_feature_matrix.shape)

# Create a DataFrame with selected features and image names as indices
selected_df = pd.DataFrame(selected_feature_matrix, index=image_names)

# Create a DataFrame with selected features and image names as columns
column_headers = [f"{key}" + str(i) for i in range(selected_feature_matrix.shape[1])]
column_headers = ["image"] + ["label"] + column_headers

# Create a DataFrame with selected features and image names as columns
selected_df = pd.DataFrame(selected_feature_matrix, columns=column_headers[2:])

# Insert the "image" column at the beginning
selected_df.insert(0, "image", image_names)

# Insert a placeholder column for the "label" column
selected_df.insert(1, "label", "")

# Assign the target variable values to the "label" column
selected_df["label"] = target_var

# Remove the target_var column from the end of the DataFrame
selected_df = selected_df.iloc[:, :-1]

# Save the DataFrame to a CSV file
selected_df.to_csv(f'{key}.csv', index=False)
