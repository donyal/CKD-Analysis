import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Code from part 2 necessary to create feature importances visualization

# Load dataset
data = pd.read_csv("output.csv")

# Separate target variable from features
X = data.drop("classification", axis=1)
y = data["classification"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Oversample the minority class using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)

# Define parameter grids
dt_param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

knn_pipeline = Pipeline([
    ('knn', KNeighborsClassifier())
])

knn_param_grid = {
    'knn__n_neighbors': [1, 3, 5],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# Perform random search to find best hyperparameters
dt = DecisionTreeClassifier(random_state=42)
dt_random_search = RandomizedSearchCV(estimator=dt, param_distributions=dt_param_grid, n_iter=50, cv=5, random_state=42)
dt_random_search.fit(X_train_resampled, y_train_resampled)

rf = RandomForestClassifier(random_state=42)
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, n_iter=50, cv=5, random_state=42)
rf_random_search.fit(X_train_resampled, y_train_resampled)

knn_grid_search = GridSearchCV(knn_pipeline, knn_param_grid, cv=5, scoring='accuracy')
knn_grid_search.fit(X_train_resampled, y_train_resampled)

# Train models with best hyperparameters
dt_best_params = dt_random_search.best_params_
best_dt = DecisionTreeClassifier(**dt_best_params, random_state=42)
best_dt.fit(X_train_resampled, y_train_resampled)

rf_best_params = rf_random_search.best_params_
best_rf = RandomForestClassifier(**rf_best_params, random_state=42)
best_rf.fit(X_train_resampled, y_train_resampled)

knn_best_model = knn_grid_search.best_estimator_

# Get feature names
features = X_train.columns


# Code for the feature importances comparison plot --------------------------------------------

# Perform permutation feature importance for the KNN model
knn_result = permutation_importance(knn_best_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
knn_importances = knn_result.importances_mean
knn_std = knn_result.importances_std

# Combine feature importances into a single DataFrame
importances_data = pd.DataFrame({
    'Feature': features,
    'Decision Tree': best_dt.feature_importances_,
    'Random Forest': best_rf.feature_importances_,
    'KNN': knn_importances
})

# Sort the features by their average importance across all models
avg_importances = importances_data[['Decision Tree', 'Random Forest', 'KNN']].mean(axis=1)
importances_data['Average Importance'] = avg_importances
importances_data.sort_values('Average Importance', ascending=False, inplace=True)

# Plot the combined feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importances Comparison")

x = np.arange(len(features))
bar_width = 0.2

plt.bar(x - bar_width, importances_data['Decision Tree'], width=bar_width, label='Decision Tree')
plt.bar(x, importances_data['Random Forest'], width=bar_width, label='Random Forest')
plt.bar(x + bar_width, importances_data['KNN'], width=bar_width, label='KNN')

plt.xticks(x, importances_data['Feature'], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.legend()
plt.tight_layout()
plt.show()
# End of code for the feature importances comparison plot -------------------------------------

# Part 1 code needed to create Heatmap visualization

# Load dataset
df = pd.read_csv("original.csv")
df = df.drop(columns='id')

# Instantiate the encoder
encoder = LabelEncoder()

# Convert 'pcv', 'wc', and 'rc' to numeric, coerce errors to NaN
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

# Fill null values in numeric columns ('pcv', 'wc', 'rc') with their mean
df['pcv'] = df['pcv'].fillna(df['pcv'].mean())
df['wc'] = df['wc'].fillna(df['wc'].mean())
df['rc'] = df['rc'].fillna(df['rc'].mean())

# Fill null values in additional numeric columns
additional_numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']
for column in additional_numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Fill null values in categorical columns with the most frequent value (mode)
categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Perform winsorization to remove outliers at the 5th and 95th percentiles
numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for column in numeric_columns:
    q1 = df[column].quantile(0.05)
    q3 = df[column].quantile(0.95)
    df[column] = df[column].mask(df[column] < q1, q1)
    df[column] = df[column].mask(df[column] > q3, q3)

# Identify non-numeric columns and apply the encoder
non_numeric_columns = df.select_dtypes(include=['object']).columns
for column in non_numeric_columns:
    df[column] = encoder.fit_transform(df[column])

# Separate features and target
features = df.drop(columns='classification')
target = df['classification']

# Perform RFE
model = LogisticRegression(solver='saga', max_iter=7000)
rfe = RFE(model)
fit = rfe.fit(features, target)



# NEW CODE RELATED TO HEATMAP ----------------------------

# Features present in the cleaned dataset
selected_features = ['al', 'su', 'rbc', 'pc', 'pcc', 'sc', 'hemo', 'rc', 'htn', 'dm', 'appet', 'pe']

# Features that were removed
dropped_features = ['age', 'bp', 'sg', 'bgr', 'bu', 'sod', 'pot', 'pcv', 'wc', 'cad', 'ane']

# Create dataframes with selected and dropped features
selected_features_df = features[selected_features]
dropped_features_df = features[dropped_features]
combined_df = pd.concat([selected_features_df, dropped_features_df], axis=1)

# Calculate correlation matrix between selected and dropped features
correlation_matrix = combined_df.corr().iloc[:len(selected_features), len(selected_features):]

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f',
            xticklabels=dropped_features,
            yticklabels=selected_features)
plt.title('Correlation Heatmap: Selected Features vs Dropped Features')
plt.xlabel('Dropped Features')
plt.ylabel('Selected Features')
plt.tight_layout()
plt.show()

# END OF HEATMAP CODE ----------------------------------------


# Random Forest Chart from Part 2 code

# Load dataset
data = pd.read_csv("output.csv")

# Encode non-numeric features
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Separate target variable from features
X = data.drop("classification", axis=1)
y = data["classification"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Calculate feature importances
importances = rf.feature_importances_
feature_names = X.columns
sorted_indices = importances.argsort()[::-1]

# Plot feature importances
plt.figure(figsize=(10, 8))
plt.bar(feature_names[sorted_indices], importances[sorted_indices])
plt.xticks(rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest - Feature Importances')
plt.tight_layout()
plt.show()

# END OF Random Forest Chart from Part 2 code

# Code from Part 1 for the Winsorization Box Plots

# Load dataset
df = pd.read_csv("original.csv")
df = df.drop(columns='id')

# Convert 'pcv', 'wc', and 'rc' to numeric, coerce errors to NaN
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

# Fill null values in numeric columns ('pcv', 'wc', 'rc') with their mean
df['pcv'] = df['pcv'].fillna(df['pcv'].mean())
df['wc'] = df['wc'].fillna(df['wc'].mean())
df['rc'] = df['rc'].fillna(df['rc'].mean())

# Additional numeric columns that need null values filled
additional_numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']
for column in additional_numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Fill null values in categorical columns with the most frequent value (mode)
categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Verify all null values are filled
null_values_final_check = df.isnull().sum()
print(null_values_final_check)

# Define Winsorization function
def winsorize_column(data, column, lower_percentile=0.05, upper_percentile=0.95):
    lower_limit = data[column].quantile(lower_percentile)
    upper_limit = data[column].quantile(upper_percentile)
    data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)
    return data

# Perform Winsorization
numeric_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for column in numeric_columns:
    df = winsorize_column(df, column)

# Adjust boxplot limits
plt.figure(figsize=(15, 10))
sns.boxplot(data=df[numeric_columns])
plt.title('Boxplot of Numeric Columns Before Winsorization')
plt.xticks(rotation=45)
plt.ylim(0, 160)  # Adjust the y-axis limit
plt.yticks(range(0, 161, 15))  # Set y-axis tick intervals to 15
plt.show()

# Visualize outliers after Winsorization with adjusted axis limits
plt.figure(figsize=(15, 10))
sns.boxplot(data=df[numeric_columns])
plt.title('Boxplot of Numeric Columns After Winsorization')
plt.xticks(rotation=45)
plt.ylim(0, 160)  # Adjust the y-axis limit
plt.yticks(range(0, 161, 15))  # Set y-axis tick intervals to 15
plt.show()