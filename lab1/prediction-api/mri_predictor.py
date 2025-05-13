# MRI Classifier logic will go here 

import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import os
import sys # Added import
from typing import Union, Dict # Added for older Python compatibility

# Define the slicing for feature reduction
# This corresponds to taking every 4th voxel in each dimension
# and the first (and only, for these MRI scans) time point/channel.
FOURTH_SLICE = pd.IndexSlice[::4, ::4, ::4, 0]

class MRIClassifier:
    def __init__(self, model_path='knn_model.pkl'):
        """
        Initializes the MRIClassifier by loading the pre-trained KNN model.
        Args:
            model_path (str): Path to the saved .pkl model file.
        """
        try:
            # Ensure the model path is correct, especially when running in Docker
            # If model_path is just 'knn_model.pkl', it assumes it's in the WORKDIR
            if not os.path.exists(model_path):
                # Attempt to find it relative to this script's directory
                script_dir = os.path.dirname(__file__)
                model_path_alt = os.path.join(script_dir, model_path)
                if os.path.exists(model_path_alt):
                    model_path = model_path_alt
                else:
                    raise FileNotFoundError(f"Model file not found at {model_path} or {model_path_alt}")
            
            self.model = joblib.load(model_path)
            print(f"Model {model_path} loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            self.model = None
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")
            self.model = None

    def _preprocess_nifti_data(self, image_path: str) -> Union[np.ndarray, None]:
        """
        Loads a single NIfTI image, extracts its data, applies a predefined slice,
        and returns the processed 3D image array.
        Adapted from preprocess.py.
        """
        try:
            img_object = nib.load(image_path)
            img_data = img_object.get_fdata()

            if img_data.ndim == 3: # If truly 3D, add a channel dimension for FOURTH_SLICE
                img_data = img_data[..., np.newaxis]
            
            if img_data.ndim == 4:
                processed_image_data = img_data[FOURTH_SLICE]
            else:
                print(f"Error: Image at {image_path} has an unexpected number of dimensions: {img_data.ndim} after potential modification.")
                return None
            
            # Ensure the output is a 3D array as expected by the model after flattening
            if processed_image_data.ndim != 3:
                 # This might happen if the slice was too aggressive or input was unexpected
                 print(f"Error: Processed image data has {processed_image_data.ndim} dimensions, expected 3.")
                 return None

            return processed_image_data

        except FileNotFoundError:
            print(f"Error: The NIfTI file was not found at {image_path}")
            return None
        except Exception as e:
            print(f"An error occurred while processing NIfTI file {image_path}: {e}")
            return None

    def predict_single_image(self, image_path: str) -> Union[Dict, None]:
        """
        Predicts the class for a single NIfTI image.
        Args:
            image_path (str): Path to the NIfTI image file.
        Returns:
            Dict: A dictionary containing the prediction result {'prediction': class_label} 
                  or {'error': message} if prediction fails.
                  Returns None if the model is not loaded.
        """
        if self.model is None:
            print("Error: Model is not loaded. Cannot predict.")
            return {'error': 'Model not loaded'}

        processed_data_3d = self._preprocess_nifti_data(image_path)

        if processed_data_3d is None:
            return {'error': f'Failed to preprocess image {image_path}'}

        try:
            # Flatten the 3D array to 1D for the model
            # The model (pipeline) expects a 2D array [n_samples, n_features]
            flattened_data = processed_data_3d.reshape(1, -1) # 1 sample, infer features
            
            prediction = self.model.predict(flattened_data)
            
            # Assuming prediction is an array like [0] or [1]
            predicted_class = int(prediction[0]) 
            
            return {'prediction': predicted_class}

        except Exception as e:
            print(f"Error during prediction for {image_path}: {e}")
            return {'error': f'Prediction failed: {e}'}

if __name__ == '__main__':
    print("Initializing MRIClassifier...")
    # Determine the model file path
    # Default is 'knn_model.pkl'.
    # Check CWD, script's directory, and parent of script's directory.
    model_file_name = 'knn_model.pkl'
    
    # Get the absolute path of the directory containing this script
    # Fallback to CWD if __file__ is not defined (e.g., in some interactive environments)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    paths_to_check = [
        model_file_name,                                  # 1. Current working directory
        os.path.join(script_dir, model_file_name),        # 2. Script's own directory
        os.path.join(script_dir, '..', model_file_name)   # 3. Parent of script's directory
    ]
    
    model_file_path = None
    for path_option in paths_to_check:
        # Normalize path for consistent checking
        normalized_path_option = os.path.normpath(path_option)
        if os.path.exists(normalized_path_option):
            model_file_path = normalized_path_option
            print(f"Found model at: {model_file_path}")
            break
            
    if model_file_path is None:
        print(f"Error: Model file '{model_file_name}' not found.")
        print(f"Looked in CWD, script directory ('{script_dir}'), and its parent.")
        print(f"Please ensure '{model_file_name}' is accessible in one of these locations.")
        exit(1)

    classifier = MRIClassifier(model_path=model_file_path)

    if classifier.model is None:
        # MRIClassifier.__init__ already prints detailed errors if model loading fails.
        print("MRIClassifier model could not be loaded (see errors above). Exiting.")
        exit(1)

    # Check if a command-line argument (image path) is provided
    if len(sys.argv) > 1:
        image_to_test = sys.argv[1]
        print(f"\nUser provided image for prediction: '{image_to_test}'")

        if not os.path.exists(image_to_test):
            print(f"Error: The provided image file was not found at '{image_to_test}'")
            print("Please provide a valid path to a NIfTI image file.")
        else:
            print(f"Attempting prediction for: {image_to_test}")
            result = classifier.predict_single_image(image_to_test)
            
            if result:
                if 'error' in result:
                    print(f"Prediction Error for '{image_to_test}': {result['error']}")
                else:
                    print(f"Prediction Result for '{image_to_test}': {result}")
            else:
                # This path might be hit if predict_single_image itself returns None
                # which can happen if _preprocess_nifti_data returns None and it's not caught to return an error dict.
                # The current _preprocess_nifti_data returns None on errors, and predict_single_image
                # catches this and returns {'error': ...}. So this path should be less common.
                print(f"Prediction for '{image_to_test}' returned an unexpected value (e.g., None). Review logs.")
    else:
        # --- Fallback to default test logic ---
        print("\nNo specific image path provided via command-line argument.")
        print("Running default tests with a dummy NIfTI image...")

        # Dummy image parameters
        dummy_image_shape = (256, 256, 128, 1) # A typical shape, last dim is channel/time
        dummy_data = np.random.rand(*dummy_image_shape).astype(np.float32)
        dummy_affine = np.eye(4) # Standard identity affine matrix
        dummy_img_obj = nib.Nifti1Image(dummy_data, dummy_affine)
        dummy_nifti_path = "temp_test_mri_image.nii.gz" # Name for the temporary file

        # Ensure clean state for the dummy file (remove if exists)
        if os.path.exists(dummy_nifti_path):
            try:
                os.remove(dummy_nifti_path)
            except OSError as e:
                print(f"Warning: Could not remove pre-existing dummy file '{dummy_nifti_path}': {e}")
                # Proceeding, as saving might overwrite or handle it.

        try:
            nib.save(dummy_img_obj, dummy_nifti_path)
            print(f"Successfully created dummy NIfTI image for testing at: {dummy_nifti_path}")

            print(f"\nPredicting for dummy image: {dummy_nifti_path}")
            result = classifier.predict_single_image(dummy_nifti_path)
            if result:
                if 'error' in result:
                    print(f"Dummy Image Prediction Error: {result['error']}")
                else:
                    print(f"Dummy Image Prediction Result: {result}")
            else:
                print("Dummy Image Prediction returned an unexpected value (e.g., None).")

        except Exception as e:
            print(f"An error occurred during the dummy image test (creation or prediction): {e}")
        finally:
            # Clean up the dummy NIfTI file
            if os.path.exists(dummy_nifti_path):
                try:
                    os.remove(dummy_nifti_path)
                    print(f"Successfully cleaned up dummy image: {dummy_nifti_path}")
                except OSError as e:
                    print(f"Warning: Could not clean up dummy file '{dummy_nifti_path}': {e}")

        # Test with a deliberately non-existent file
        print(f"\nPredicting for 'a_surely_non_existent_image.nii.gz' (expected to fail gracefully)")
        non_existent_result = classifier.predict_single_image("a_surely_non_existent_image.nii.gz")
        if non_existent_result and 'error' in non_existent_result:
            print(f"Handled non-existent file as expected. Error: {non_existent_result['error']}")
        elif non_existent_result:
             print(f"Non-existent file test returned an unexpected result: {non_existent_result}")
        else:
            print(f"Non-existent file test did not return a dictionary as expected (e.g., got None).")

    print("\nMRIClassifier script finished.") 