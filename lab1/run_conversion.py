import os
from preprocess import convert_img_hdr_pair_to_nii_gz # Make sure preprocess.py is in the same directory

def main():
    # --- Configuration: Set your file paths here ---

    # Set the base directory where your RAW NIfTI files are located
    # This should be the directory containing 'mpr-1.nifti.img' and 'mpr-1.nifti.hdr'
    # Based on your screenshot, it looks like they might be in a 'RAW' subdirectory.
    # Please adjust this path if your 'mpr-1.nifti.img' etc. are elsewhere.
    # For example, if 'Assignment2' contains a folder 'OAS1_0001_MR1/RAW/' which holds the images:
    # base_input_dir = "/Users/loesvanvoorden/Library/CloudStorage/OneDrive-Personal/MSc JADS/JADS year 1/2 Advanced Data Architectures/Assignment2/OAS1_0001_MR1/RAW/"
    
    # Let's assume for now the files are directly in the Assignment2 directory for simplicity
    # If they are in a 'RAW' subfolder:
    # raw_folder_path = "/Users/loesvanvoorden/Library/CloudStorage/OneDrive-Personal/MSc JADS/JADS year 1/2 Advanced Data Architectures/Assignment2/RAW/"
    # If you are running this script from the Assignment2 folder itself, and RAW is a subfolder:

    # Name of the .img file you want to convert
    img_file_name_to_convert = "mpr-1.nifti.img"

    # Path to the input .img file
    input_img_path = img_file_name_to_convert

    # Define where you want to save the converted .nii.gz file and its name
    # It's good practice to save it in a place you can easily find, maybe an 'converted_nifti' folder
    # or directly in your Assignment2 folder for now.
    output_directory = "/Users/loesvanvoorden/Library/CloudStorage/OneDrive-Personal/MSc JADS/JADS year 1/2 Advanced Data Architectures/Assignment2/"
    
    # Let's create an 'converted_nifti' subfolder if it doesn't exist
    converted_output_dir = os.path.join(output_directory, "converted_nifti_files")
    if not os.path.exists(converted_output_dir):
        os.makedirs(converted_output_dir)
        print(f"Created directory: {converted_output_dir}")

    base_name_for_output = os.path.splitext(os.path.splitext(img_file_name_to_convert)[0])[0] # Gets 'mpr-1' from 'mpr-1.nifti.img'
    output_nii_gz_filename = base_name_for_output + "_converted.nii.gz" # e.g., "mpr-1_converted.nii.gz"
    output_nii_gz_full_path = os.path.join(converted_output_dir, output_nii_gz_filename)

    # --- End Configuration ---

    print(f"Attempting to convert: {input_img_path}")
    print(f"The corresponding .hdr file is expected to be in the same directory")
    print(f"Output will be saved to: {output_nii_gz_full_path}")

    if os.path.exists(input_img_path):
        success = convert_img_hdr_pair_to_nii_gz(input_img_path, output_nii_gz_full_path)
        if success:
            print(f"\nConversion successful!")
            print(f"Converted file saved at: {output_nii_gz_full_path}")
            print("\nYou can now use this .nii.gz file with your prediction API.")
        else:
            print(f"\nConversion failed for {input_img_path}.")
            print("Please check the error messages above from the conversion function.")
    else:
        print(f"\nError: Input .img file not found at '{input_img_path}'. Please check the path.")

if __name__ == "__main__":
    main()
