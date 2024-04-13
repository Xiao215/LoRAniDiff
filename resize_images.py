from PIL import Image
import os

def resize_images(image_dir):
  """
  Resizes all images in a directory to 64x64 pixels and saves them in the same directory.

  Args:
      image_dir: Path to the directory containing the images.
  """
  for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # Check for common image extensions
      image_path = os.path.join(image_dir, filename)
      try:
        img = Image.open(image_path)
        img = img.resize((NEW_SIZE, NEW_SIZE)) 
        img.save(image_path)  # Save the resized image
        print(f"Resized {filename}")
      except Exception as e:
        print(f"Error resizing {filename}: {e}")

    

IMAGE_DIR = './image/downloads32/'
NEW_SIZE = 32

resize_images(IMAGE_DIR)