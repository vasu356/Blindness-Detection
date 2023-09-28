from PIL import Image
import os

# Define the input and output directories
input_directory = 'C:\\Users\\KIIT\\Desktop\\Code Clause Projects\\Task 3\\images\\resized test 15'
output_directory = 'C:\\Users\\KIIT\\Desktop\\Code Clause Projects\\Task 3\\images\\test_resized'
target_size = (224, 224)  # Adjust this size as need

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Resize and save images
for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter for image files
        try:
            img = Image.open(os.path.join(input_directory, filename))
            img = img.resize(target_size, Image.BILINEAR)  # Use 'BILINEAR' for resizing
            img.save(os.path.join(output_directory, filename))
            print(f'Resized and saved: {filename}')
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')
