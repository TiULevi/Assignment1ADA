import os
from preprocess import convert_img_hdr_pair_to_nii_gz # Assumes preprocess.py is in the same directory (data_processing)

def main():
    # --- Configuration: Set your file paths here ---

    # This script assumes it is in lab1/data_processing/
    # Raw data is expected in lab1/data/raw/
    # Converted data will be saved to lab1/data/converted/

    # Relative path from this script (in data_processing) to the lab1 directory
    lab1_base_dir = os.path.join(os.path.dirname(__file__), "..") 
    lab1_base_dir = os.path.normpath(lab1_base_dir) # Normalize path (e.g. remove ../)

    # Path to the raw data directory
    raw_data_dir = os.path.join(lab1_base_dir, "data", "raw")

    # Path to the output directory for converted files
    converted_output_dir = os.path.join(lab1_base_dir, "data", "converted")
    
    # Name of the .img file you want to convert (should be in raw_data_dir)
    img_file_name_to_convert = "mpr-1.nifti.img" # Example file name

    # Full path to the input .img file
    input_img_path = os.path.join(raw_data_dir, img_file_name_to_convert)

    # Create the output directory if it doesn't exist
    if not os.path.exists(converted_output_dir):
        os.makedirs(converted_output_dir)
        print(f"Created directory: {converted_output_dir}")

    # Define the output file name and full path
    base_name_for_output = os.path.splitext(os.path.splitext(img_file_name_to_convert)[0])[0]
    output_nii_gz_filename = base_name_for_output + "_converted.nii.gz"
    output_nii_gz_full_path = os.path.join(converted_output_dir, output_nii_gz_filename)

    # --- End Configuration ---

    print(f"Source file: {input_img_path}")
    print(f"Target converted file: {output_nii_gz_full_path}")

    if not os.path.exists(raw_data_dir):
        print(f"Error: Raw data directory not found at '{raw_data_dir}'. Please create it and place your .img/.hdr files there.")
        return

    if os.path.exists(input_img_path):
        print(f"Attempting to convert: {input_img_path}")
        print(f"The corresponding .hdr file is expected to be in the same directory: {raw_data_dir}")
        success = convert_img_hdr_pair_to_nii_gz(input_img_path, output_nii_gz_full_path)
        if success:
            print(f"\nConversion successful!")
            print(f"Converted file saved at: {output_nii_gz_full_path}")
        else:
            print(f"\nConversion failed for {input_img_path}.")
            print("Please check the error messages from the conversion function in preprocess.py.")
    else:
        print(f"\nError: Input .img file not found at '{input_img_path}'.")
        print(f"Please ensure the file '{img_file_name_to_convert}' exists in '{raw_data_dir}'.")

if __name__ == "__main__":
    main() 