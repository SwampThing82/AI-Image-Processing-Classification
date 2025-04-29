import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2


model = MobileNetV2(weights="imagenet")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for a given image and model."""
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def add_black_box(img, top_left, bottom_right):
    """Add a black box to an image."""
    img_copy = img.copy()
    cv2.rectangle(img_copy, top_left, bottom_right, color=(0, 0, 0), thickness=-1)
    return img_copy

def blur_region(img, top_left, bottom_right, ksize=(25, 25)):
    """Blur a region of an image."""
    img_copy = img.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    region = img_copy[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(region, ksize, 0)
    img_copy[y1:y2, x1:x2] = blurred
    return img_copy

def add_noise_patch(img, top_left, bottom_right):
    """Add random noise to a region of an image."""
    img_copy = img.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    h, w = y2 - y1, x2 - x1
    noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img_copy[y1:y2, x1:x2] = noise
    return img_copy    

def classify_image(image_path):
    """Classify an image and display the predictions."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        
        img_array = image.img_to_array(img)
        
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

     # Grad-CAM heatmap
        last_conv_layer_name = "Conv_1"
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Load original image
        img = image.load_img(image_path)
        img = image.img_to_array(img)

    # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

    # Apply heatmap onto original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

    # Plot the result
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img.astype("uint8"))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Grad-CAM")
        plt.imshow(superimposed_img.astype("uint8"))
        plt.axis('off')

        plt.savefig("gradcam_result.png")
        print("Grad-CAM image saved as gradcam-result.png")

        # Load the original image
        img_original = image.load_img(image_path)
        img_original = image.img_to_array(img_original)

        # Set coordinates for occlusion
        box_width, box_height = 60, 180
        center_x, center_y = 100, 112  # (adjusting to cover whole face)

        top_left = (center_x - box_width // 2, center_y - box_height // 2)
        bottom_right = (top_left[0] + box_width, top_left[1] + box_height)

        # Apply Black Box
        img_black_box = add_black_box(img_original, top_left, bottom_right)
        cv2.imwrite("black_box_occlusion.jpg", cv2.cvtColor(img_black_box, cv2.COLOR_RGB2BGR))

        # Apply Blur
        img_blurred = blur_region(img_original, top_left, bottom_right)
        cv2.imwrite("blurred_occlusion.jpg", cv2.cvtColor(img_blurred, cv2.COLOR_RGB2BGR))

        # Apply Random Noise
        img_noisy = add_noise_patch(img_original, top_left, bottom_right)
        cv2.imwrite("noise_patch_occlusion.jpg", cv2.cvtColor(img_noisy, cv2.COLOR_RGB2BGR))

        print("Saved three occluded images: black_box_occlusion.jpg, blurred_occlusion.jpg, noise_patch_occlusion.jpg")

        # --- Now classify each occluded image ---
        occluded_images = {
            "Black Box": img_black_box,
            "Blurred": img_blurred,
            "Noisy": img_noisy,
        }

        for occlusion_type, occluded_img in occluded_images.items():
            # Preprocess occluded image
            occluded_img_array = preprocess_input(occluded_img)
            occluded_img_array = np.expand_dims(occluded_img_array, axis=0)

            # Make prediction
            occluded_predictions = model.predict(occluded_img_array)
            decoded_occluded_predictions = decode_predictions(occluded_predictions, top=3)[0]

            print(f"\nTop-3 Predictions for {occlusion_type} Occlusion:")
            for i, (imagenet_id, label, score) in enumerate(decoded_occluded_predictions):
                print(f"{i + 1}: {label} ({score:.2f})")

    except Exception as e:
        print(f"Error processing image: {e}")



if __name__ == "__main__":
    image_path = "basic_murphy.jpg"  
    classify_image(image_path)