import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def load_and_preprocess_data(pickle_path="Full_Image_data.pkl"):
    """Loads and preprocesses the data from the pickle file."""
    try:
        data = pd.read_pickle(pickle_path)
    except FileNotFoundError:
        print(f"Error: The file {pickle_path} was not found. Please ensure preprocess.py has been run or the file exists.")
        return None, None
    
    # Also drop rows where Images or Group are missing
    # 'Images' column is critical for features, 'Group' for target
    if 'Images' not in data.columns or 'Group' not in data.columns:
        print("Error: 'Images' or 'Group' column not found. Cannot proceed.")
        return None, None
    data = data.dropna(subset=['Images', 'Group'])

    # Reset index to align everything properly
    data = data.reset_index(drop=True)

    # Map all possible classes for Group
    # Original notebook: {'Nondemented': 0, 'Demented': 1, 'Converted': 2}
    # We assume 'Group' might already be numeric from the pickle or was object
    if 'Group' in data.columns and data['Group'].dtype == 'object':
        group_map = {'Nondemented': 0, 'Demented': 1, 'Converted': 2}
        data['Group'] = data['Group'].map(group_map)
    
    # Ensure 'Group' is integer type if it's not already
    if 'Group' in data.columns:
        data['Group'] = data['Group'].astype(int)


    # Exclude group 2 (Converted) from the dataset, as done for "without SMOTE"
    if 'Group' in data.columns:
        data = data[data['Group'] != 2]
        data = data.reset_index(drop=True) # Reset index again after filtering
    else:
        print("Warning: 'Group' column not found for filtering.")
        # Or handle as an error depending on requirements
        
    if data.empty:
        print("Error: Data is empty after preprocessing and filtering. Cannot proceed.")
        return None, None

    # Flatten image data
    # The 'Images' column should contain NumPy arrays (pre-sliced by preprocess_single_nifti logic originally)
    # Each image is expected to be 3D (e.g., 64x64x32 from the 'fourth' slice)
    try:
        # Check if images are already numpy arrays and have 3 dimensions
        if not data['Images'].apply(lambda x: isinstance(x, np.ndarray) and x.ndim == 3).all():
            print("Error: Not all entries in 'Images' column are 3D NumPy arrays.")
            # Attempt to describe the problematic entries
            for i, item in enumerate(data['Images']):
                if not (isinstance(item, np.ndarray) and item.ndim == 3):
                    print(f"Problematic entry at index {i}: type={type(item)}, ndim={item.ndim if isinstance(item, np.ndarray) else 'N/A'}")
                    if isinstance(item, np.ndarray):
                        print(f"Shape: {item.shape}")
            return None, None

        # Convert list of flattened images into a DataFrame
        # Each image (e.g., 64x64x32) is reshaped to a 1D array (131072 features)
        image_flat_list = [img.reshape(-1) for img in data['Images']]
        X = pd.DataFrame(image_flat_list)
        # Ensure column names are strings for scikit-learn compatibility
        X.columns = X.columns.astype(str)
    except AttributeError:
        print("Error: Could not reshape images. Ensure 'Images' column contains NumPy arrays.")
        return None, None
    except Exception as e:
        print(f"Error during image flattening: {e}")
        return None, None
        
    if 'Group' not in data.columns:
        print("Error: Target 'Group' column is missing after processing.")
        return None, None
        
    y = data['Group']
    
    if X.empty or y.empty:
        print("Error: Feature set X or target y is empty before train-test split.")
        return None, None

    return X, y

def main():
    X, y = load_and_preprocess_data()

    if X is None or y is None:
        print("Exiting due to data loading or preprocessing errors.")
        return

    if X.shape[0] != y.shape[0]:
        print(f"Error: X and y have mismatched number of samples: X({X.shape[0]}), y({y.shape[0]})")
        return
        
    if y.nunique() < 2:
        print(f"Error: Target variable 'y' has less than 2 unique classes after preprocessing: {y.unique()}. Classification requires at least 2 classes.")
        return

    # Split data: test_size=0.3, random_state=42, stratify=y (as per notebook for KNN "without SMOTE")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Error during train_test_split (likely due to insufficient samples for stratification): {e}")
        print(f"Number of samples: {X.shape[0]}, Unique classes in y: {y.value_counts().to_dict()}")
        return


    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")

    # Define the pipeline for KNN - StandardScaler will be included here for GridSearchCV
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('knn', KNeighborsClassifier())
    ])

    # Define hyperparameter grid for GridSearchCV (as per user snippet)
    knn_param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['minkowski', 'euclidean', 'manhattan']
    }

    # Define StratifiedKFold (n_splits=3 as per user snippet for non-SMOTE KNN)
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Perform GridSearchCV for KNN
    print("\nPerforming GridSearchCV for KNN (without SMOTE)...")
    knn_grid_search = GridSearchCV(
        knn_pipeline, knn_param_grid, cv=stratified_kfold, scoring='accuracy', n_jobs=3 # n_jobs adjusted as per snippet
    )
    knn_grid_search.fit(X_train, y_train)

    print("\nBest KNN parameters:", knn_grid_search.best_params_)
    print("Best KNN cross-validated accuracy:", knn_grid_search.best_score_)

    # Get the best estimator
    best_knn_model = knn_grid_search.best_estimator_

    # Make predictions on the training set
    y_train_pred_knn = best_knn_model.predict(X_train)

    # Print classification report for the training set
    print("\nKNN Training Set Classification Report:")
    print(classification_report(y_train, y_train_pred_knn))

    # Make predictions on the test set
    y_test_pred_knn = best_knn_model.predict(X_test)

    # Print classification report for the test set
    print("\nKNN Test Set Classification Report:")
    print(classification_report(y_test, y_test_pred_knn))

    # Save the best KNN model
    model_filename = 'knn_model.pkl'
    joblib.dump(best_knn_model, model_filename)
    print(f"\nBest KNN model saved to {model_filename}")

if __name__ == "__main__":
    main() 