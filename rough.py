import os
from PIL import Image
import matplotlib.pyplot as plt

def resize_image(input_path, size):
    # Open the image file
    with Image.open(input_path) as img:
        # Resize the image
        resized_img = img.resize(size)
        
        # Display the original and resized images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        
        ax1.set_title('Original Image')
        ax1.imshow(img)
        ax1.axis('off')
        
        ax2.set_title('Resized Image ({}x{})'.format(*size))
        ax2.imshow(resized_img)
        ax2.axis('off')
        
        plt.show()

# Example usage
input_image_path = os.path.join(os.getcwd(), 'small_data', 'test', 'Positive', '00011.jpg')

# Specify the desired size
target_size = (64, 64)

# Resize the image
resize_image(input_image_path, target_size)