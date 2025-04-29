from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import cv2
import numpy as np

def apply_blur_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        plt.imshow(img_blurred)
        plt.axis('off')
        plt.savefig("blurred_image.png")
        print("Processed image saved as 'blurred_image.png'.")

    except Exception as e:
        print(f"Error processing image: {e}")

def apply_advanced_comic_effect(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 1: Apply bilateral filter (smooth colors but keep edges sharp)
    color_img = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=200)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Step 3: Apply median blur to smooth the grayscale image
    gray_blur = cv2.medianBlur(gray, 7)

    # Step 4: Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(
        gray_blur, 
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    # Step 5: Combine color image with edge mask
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    comic_img = cv2.bitwise_and(color_img, edges_colored)

    # Step 6: Save the final result
    comic_img_bgr = cv2.cvtColor(comic_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV save
    cv2.imwrite("comic_effect_advanced.png", comic_img_bgr)
    print("Saved 'comic_effect_advanced.png'.")

def apply_filters_and_save(image_path):
    try:
        img = Image.open(image_path)

        # Apply Edge Detection
        img_edges = img.filter(ImageFilter.FIND_EDGES)
        img_edges.save("edges_detected.png")
        print("Saved 'edges_detected.png'.")

        # Apply Sharpening
        img_sharpened = img.filter(ImageFilter.SHARPEN)
        img_sharpened.save("sharpened_image.png")
        print("Saved 'sharpened_image.png'.")

        # Apply Gaussian Blur
        img_blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
        img_blurred.save("gaussian_blur.png")
        print("Saved 'gaussian_blur.png'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "basic_murphy.jpg"  # Replace with the path to your image file
    apply_blur_filter(image_path)
    apply_advanced_comic_effect(image_path)
    apply_filters_and_save(image_path)