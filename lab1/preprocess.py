import nibabel as nib
import pandas as pd
import numpy as np
import os # Added os import for path manipulation in new function
from typing import Union # Added for older Python compatibility

# Define the slicing for feature reduction
# This corresponds to taking every 4th voxel in each dimension
# and the first (and only, for these MRI scans) time point/channel.
FOURTH_SLICE = pd.IndexSlice[::4, ::4, ::4, 0]

def preprocess_single_nifti(image_path: str) -> Union[np.ndarray, None]: # Updated type hint
    """
    Loads a single NIfTI image, extracts its data, applies a predefined slice 
    for feature reduction, and returns the processed 3D image array.

    Args:
        image_path (str): The file path to the NIfTI image 
                          (e.g., '.img', '.nii', '.nii.gz').

    Returns:
        Union[np.ndarray, None]: The processed 3D image data as a NumPy array.
                                 Returns None if loading or processing fails.
    """
    try:
        # Load the NIfTI image
        img_object = nib.load(image_path)
        
        # Get the image data as a NumPy array
        # get_fdata() loads the data and ensures it's in a standard orientation (RAS+)
        # and applies any scaling from the header.
        img_data = img_object.get_fdata()
        
        # Apply the predefined slice for feature reduction
        # This reduces the image dimensions by a factor of 4 in each spatial dimension.
        # Assuming the image is 4D (x, y, z, time/channel) and we take the first channel/timepoint.
        # If images are already 3D, the last part of the slice (0) might need adjustment
        # or ensure img_data is 4D. For typical structural MRI, it might be (x,y,z) or (x,y,z,1).
        # nibabel often loads them as 4D even if the last dim is 1.
        if img_data.ndim == 3:
            # If 3D, we might need to temporarily make it 4D for the slicer or adjust slicer
            # For simplicity, let's assume we can slice the 3D array if it's already reduced in the 4th dim
            # Or, if the slicer is (::4, ::4, ::4), it would work directly on 3D
            # Given the original FOURTH_SLICE, let's ensure it's at least 3D with a possible 4th dim.
            # The original notebook context implied the data was 4D before slicing.
            # If the raw .img files are 3D, then FOURTH_SLICE should be pd.IndexSlice[::4, ::4, ::4]
            # However, sticking to the provided FOURTH_SLICE:
            if img_data.ndim == 3: # If truly 3D, add a channel dimension
                 img_data = img_data[..., np.newaxis]
            processed_image_data = img_data[FOURTH_SLICE]

        elif img_data.ndim == 4:
            processed_image_data = img_data[FOURTH_SLICE]
        else:
            print(f"Error: Image at {image_path} has an unexpected number of dimensions: {img_data.ndim}")
            return None

        return processed_image_data

    except FileNotFoundError:
        print(f"Error: The file was not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return None

def convert_img_hdr_pair_to_nii_gz(img_file_path: str, output_nii_gz_path: str) -> bool:
    """
    Converts a NIfTI .img/.hdr pair to a single compressed .nii.gz file.

    Args:
        img_file_path (str): Path to the .img file of the pair.
                             The .hdr file is assumed to be in the same directory
                             with the same base name.
        output_nii_gz_path (str): Desired path for the output .nii.gz file.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        base_path, img_ext = os.path.splitext(img_file_path)
        hdr_file_path = "" # Initialize
        if img_ext.lower() == '.img':
            potential_hdr_path_1 = base_path + ".hdr"
            # Handle cases like 'filename.nifti.img' -> 'filename.nifti.hdr'
            potential_hdr_path_2 = img_file_path.lower().replace(img_ext.lower(), ".hdr")
            if os.path.exists(potential_hdr_path_1):
                hdr_file_path = potential_hdr_path_1
            elif os.path.exists(potential_hdr_path_2):
                hdr_file_path = potential_hdr_path_2
            else:
                print(f"Error: Corresponding .hdr file not found for {img_file_path} (expected near {base_path})")
                return False
        else:
            print(f"Error: Input file {img_file_path} does not seem to be a .img file based on extension.")
            return False

        if not (os.path.exists(img_file_path) and os.path.exists(hdr_file_path)):
             print(f"Error: Input image or its inferred header not found. Searched for .img at {img_file_path} and .hdr at {hdr_file_path}")
             return False

        print(f"Attempting to load .img file: {img_file_path} (using .hdr: {hdr_file_path})")
        loaded_image_obj = nib.load(img_file_path)

        # Create Nifti1Image using data, affine, and the original Nifti1PairHeader.
        # The Nifti1Image constructor will create a new Nifti1Header from the Nifti1PairHeader.
        nifti1_single_file_obj = nib.Nifti1Image(loaded_image_obj.get_fdata(dtype=np.float32),
                                                 loaded_image_obj.affine,
                                                 loaded_image_obj.header)
        
        # Now, ensure sform and qform are correctly set on the *new* header of nifti1_single_file_obj.
        # Access sform/qform matrices and their codes from the *original* loaded_image_obj.header.

        # SFORM handling
        sform_matrix = loaded_image_obj.header.get_sform() # get_sform() method should be available
        sform_code_val = 0 # Default to 'unknown'
        try:
            code_from_hdr = loaded_image_obj.header['sform_code'] # Access as a field
            if code_from_hdr is not None:
                sform_code_val = int(code_from_hdr)
        except (KeyError, TypeError):
            print("Warning: Original header['sform_code'] not found or None. Defaulting sform_code to 0.")
        # No explicit "AttributeError" here as direct field access should raise KeyError if missing.
        
        if sform_matrix is not None:
            nifti1_single_file_obj.header.set_sform(sform_matrix, sform_code_val)
        # else: Nifti1Image constructor and its header's from_header logic should handle defaults.

        # QFORM handling
        qform_matrix = loaded_image_obj.header.get_qform() # get_qform() method should be available
        qform_code_val = 0 # Default to 'unknown'
        try:
            code_from_hdr = loaded_image_obj.header['qform_code'] # Access as a field
            if code_from_hdr is not None:
                qform_code_val = int(code_from_hdr)
        except (KeyError, TypeError):
            print("Warning: Original header['qform_code'] not found or None. Defaulting qform_code to 0.")

        if qform_matrix is not None:
            nifti1_single_file_obj.header.set_qform(qform_matrix, qform_code_val)
        # else: Nifti1Image constructor and its header's from_header logic should handle defaults.

        nib.save(nifti1_single_file_obj, output_nii_gz_path)
        print(f"Successfully converted {img_file_path} (and its .hdr) to {output_nii_gz_path}")
        return True

    except FileNotFoundError:
        print(f"Error: One of the files (.img or .hdr) was not found during the conversion process (FileNotFoundError).")
        return False
    except Exception as e:
        print(f"An error occurred during conversion of {img_file_path}: {e}")
        return False

if __name__ == "__main__":
    # This is an example of how to use the function.
    # Replace 'path/to/your/image.img' with an actual image path.
    # Create a dummy NIFTI file for testing if you don't have one.
    
    # Example: Create a dummy NIFTI file for demonstration
    # In a real scenario, you would point to an existing NIFTI file.
    dummy_image_shape = (256, 256, 128, 1) # Example full shape before slicing
    dummy_data = np.random.rand(*dummy_image_shape).astype(np.float32)
    dummy_affine = np.eye(4)
    dummy_img_obj = nib.Nifti1Image(dummy_data, dummy_affine)
    dummy_image_path = "dummy_image.nii.gz"
    nib.save(dummy_img_obj, dummy_image_path)
    
    print(f"Attempting to preprocess: {dummy_image_path}")
    processed_img = preprocess_single_nifti(dummy_image_path)
    
    if processed_img is not None:
        print(f"Successfully processed image.")
        print(f"Shape of the original dummy image: {dummy_data.shape}")
        # After FOURTH_SLICE (::4, ::4, ::4, 0), the shape should be (64, 64, 32)
        print(f"Shape of the processed image: {processed_img.shape}") 
        
        # Expected shape: (256/4, 256/4, 128/4) = (64, 64, 32)
        expected_shape = (dummy_image_shape[0]//4, dummy_image_shape[1]//4, dummy_image_shape[2]//4)
        assert processed_img.shape == expected_shape, \
            f"Processed image shape is {processed_img.shape}, expected {expected_shape}"
        print("Shape check passed.")

    else:
        print(f"Failed to process {dummy_image_path}")

    # --- Example with a non-existent file ---
    # print("\nAttempting to preprocess a non-existent file:")
    # non_existent_file = "path/to/non_existent_image.img"
    # processed_non_existent = preprocess_single_nifti(non_existent_file)
    # if processed_non_existent is None:
    #     print("Correctly handled non-existent file.")
    
    # Clean up the dummy file
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
        print(f"Cleaned up {dummy_image_path}") 

    print("\n--- Testing convert_img_hdr_pair_to_nii_gz ---")
    # Create dummy .hdr/.img files for testing the conversion function
    # These will be Nifti1PairImage objects
    base_dummy_pair_name = "dummy_pair_image"
    dummy_img_pair_path = base_dummy_pair_name + ".img"
    dummy_hdr_pair_path = base_dummy_pair_name + ".hdr" # Nibabel will create this from Nifti1PairImage
    converted_output_path = base_dummy_pair_name + "_converted.nii.gz"

    # Sample data for the pair
    pair_data_shape = (64, 64, 32, 1) # Smaller dummy for quicker test
    pair_data = np.arange(np.prod(pair_data_shape), dtype=np.int16).reshape(pair_data_shape)
    pair_affine = np.diag([1,1,1,0])
    pair_affine[0,3] = 10
    pair_affine[1,3] = 20
    pair_affine[2,3] = 30

    # Create a Nifti1PairImage (represents .hdr/.img pair)
    # Note: Nifti1Pair header by default does not have all sform/qform info as Nifti1Image
    dummy_pair_obj = nib.Nifti1Pair(pair_data, pair_affine)
    # Set some basic header info that might be missing otherwise
    dummy_pair_obj.header.set_data_dtype(np.int16)
    dummy_pair_obj.header.set_zooms((1.0, 1.0, 1.0, 1.0))
    # Force qform and sform to be something basic for testing if not set by default
    if dummy_pair_obj.header.get_qform() is None:
        dummy_pair_obj.header.set_qform(pair_affine, code=1) # Scanner anatomica
    if dummy_pair_obj.header.get_sform() is None:
        dummy_pair_obj.header.set_sform(pair_affine, code=1) # Scanner anatomical

    nib.save(dummy_pair_obj, dummy_img_pair_path) # This saves both .img and .hdr
    print(f"Created dummy .hdr/.img pair: {dummy_hdr_pair_path}, {dummy_img_pair_path}")

    if os.path.exists(dummy_hdr_pair_path) and os.path.exists(dummy_img_pair_path):
        success = convert_img_hdr_pair_to_nii_gz(dummy_img_pair_path, converted_output_path)
        if success and os.path.exists(converted_output_path):
            print(f"Conversion test successful. Output at {converted_output_path}")
            # Optional: Load and check the converted file
            try:
                converted_loaded_obj = nib.load(converted_output_path)
                print(f"Successfully loaded converted file. Shape: {converted_loaded_obj.shape}, Data type: {converted_loaded_obj.get_data_dtype()}")
                # Check if data matches (consider potential dtype changes during save/load)
                np.testing.assert_array_almost_equal(converted_loaded_obj.get_fdata(dtype=np.float32), pair_data.astype(np.float32))
                print("Data consistency check passed.")
            except Exception as e:
                print(f"Error loading or verifying converted file: {e}")
        else:
            print("Conversion test failed or output file not created.")
    else:
        print(f"Failed to create dummy .hdr/.img pair for conversion test.")
    
    # Clean up dummy files for conversion test
    for f_path in [dummy_img_pair_path, dummy_hdr_pair_path, converted_output_path]:
        if os.path.exists(f_path):
            os.remove(f_path)
            print(f"Cleaned up {f_path}") 