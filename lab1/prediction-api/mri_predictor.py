# MRI Classifier logic will go here 

import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import os
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
    print("Testing MRIClassifier...")
    model_file_path = 'knn_model.pkl' 
    if not os.path.exists(model_file_path):
        alt_model_path = os.path.join(os.path.dirname(__file__), '..', 'knn_model.pkl')
        if os.path.exists(alt_model_path):
            model_file_path = alt_model_path
        else:
            print(f"Error: Model file {model_file_path} (and {alt_model_path}) not found.")
            exit()
            
    classifier = MRIClassifier(model_path=model_file_path)
    if classifier.model is not None:
        dummy_image_shape = (256, 256, 128, 1)
        dummy_data = np.random.rand(*dummy_image_shape).astype(np.float32)
        dummy_affine = np.eye(4)
        dummy_img_obj = nib.Nifti1Image(dummy_data, dummy_affine)
        dummy_nifti_path = "test_temp_image.nii.gz"
        if os.path.exists(dummy_nifti_path):
            os.remove(dummy_nifti_path)
        nib.save(dummy_img_obj, dummy_nifti_path)
        print(f"Created dummy NIfTI image for testing at: {dummy_nifti_path}")
        print(f"\nPredicting for dummy image: {dummy_nifti_path}")
        result = classifier.predict_single_image(dummy_nifti_path)
        if result:
            if 'error' in result:
                print(f"Prediction Error: {result['error']}")
            else:
                print(f"Prediction Result: {result}")
        else:
            print("Prediction returned None or an unexpected value.")
        print(f"\nPredicting for non_existent_image.nii.gz")
        non_existent_result = classifier.predict_single_image("non_existent_image.nii.gz")
        if non_existent_result and 'error' in non_existent_result:
            print(f"Handled non-existent file correctly: {non_existent_result['error']}")
        else:
            print(f"Non-existent file test did not behave as expected: {non_existent_result}")
        if os.path.exists(dummy_nifti_path):
            os.remove(dummy_nifti_path)
            print(f"Cleaned up {dummy_nifti_path}")
    else:
        print("MRIClassifier model not loaded. Cannot run tests.") 