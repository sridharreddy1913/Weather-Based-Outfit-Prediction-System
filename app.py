# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('ml_proj.csv', delimiter=';', encoding='ISO-8859-1')

# Encode the Weather condition
weather_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X = weather_encoder.fit_transform(data[['Weather']])

# Encode the outfit columns as targets
outfit_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
y = outfit_encoder.fit_transform(data[['Outfit1', 'Outfit2', 'Outfit3']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up a Random Forest model for multi-output classification
base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
model = MultiOutputClassifier(base_model)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__max_depth': [None, 10, 20],
    'estimator__min_samples_split': [2, 5, 10],
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Save the best model and encoders to a file
joblib.dump(best_model, 'weather_to_outfit_model.joblib')
joblib.dump(weather_encoder, 'weather_encoder.joblib')
joblib.dump(outfit_encoder, 'outfit_encoder.joblib')
print("Model and encoders saved as 'weather_to_outfit_model.joblib', 'weather_encoder.joblib', and 'outfit_encoder.joblib'")

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Decode the predictions and actual values for readability
y_pred_decoded = outfit_encoder.inverse_transform(y_pred)
y_test_decoded = outfit_encoder.inverse_transform(y_test)

# Classification report for each outfit component
for i, outfit in enumerate(['Outfit1', 'Outfit2', 'Outfit3']):
    print(f"Classification Report for {outfit}:")
    print(classification_report(y_test_decoded[:, i], y_pred_decoded[:, i], zero_division=1))

# Plot the confusion matrices for each outfit component
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, outfit in enumerate(['Outfit1', 'Outfit2', 'Outfit3']):
    ConfusionMatrixDisplay.from_predictions(y_test_decoded[:, i], y_pred_decoded[:, i], ax=axes[i], cmap='Blues')
    axes[i].set_title(f"Confusion Matrix for {outfit}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Plotting comparison of actual and predicted classes for each outfit component
for i, outfit in enumerate(['Outfit1', 'Outfit2', 'Outfit3']):
    results_df = pd.DataFrame({'Actual': y_test_decoded[:, i], 'Predicted': y_pred_decoded[:, i]}).melt(var_name='Type', value_name=outfit)

    plt.figure(figsize=(10, 6))
    sns.countplot(data=results_df, x=outfit, hue='Type', order=sorted(set(results_df[outfit].dropna())))
    plt.title(f"Comparison of Actual and Predicted Counts for {outfit}")
    plt.xlabel(outfit)
    plt.ylabel("Count")
    plt.legend(title='Class Type', loc='upper right')
    plt.xticks(rotation=45)
    plt.show()

# Plot Feature Importance for Weather features
feature_importances = best_model.estimators_[0].feature_importances_
feature_names = weather_encoder.get_feature_names_out(['Weather'])
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importances in RandomForestClassifier for Weather")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Combined multi-output classification report
print("Combined Classification Report for Weather Condition to Outfit Prediction:")
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix for combined multi-output predictions
ConfusionMatrixDisplay.from_predictions(y_test.ravel(), y_pred.ravel(), cmap='Blues')
plt.title("Combined Confusion Matrix for Weather Condition to Outfit Prediction")
plt.show()
