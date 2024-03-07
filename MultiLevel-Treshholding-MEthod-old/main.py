import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1.These lines import the necessary libraries: cv2 for computer vision operations,
# numpy for numerical operations, and
# matplotlib.pyplot for plotting.

def multilevel_thresholding(image, num_levels):
    # 2.This function takes an image and the number of threshold levels as input
    # It first converts the input image to grayscale and
    # then applies adaptive thresholding using Gaussian mean to create a binary image.
    # Calculate initial thresholds

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding using Gaussian mean
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 2.

    initial_thresholds = np.linspace(0, 255, num_levels + 2)[1:-1]

    # Refine thresholds using iterative Otsu's method
    for _ in range(10):  # You can adjust the number of iterations
        thresholds = []
        for i in range(num_levels - 1):
            region = binary_image[
                (binary_image >= initial_thresholds[i]) & (binary_image <= initial_thresholds[i + 1])
            ]
            if region.size > 0:  # Check if the region is not empty
                _, threshold = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresholds.append(threshold)

        # Update initial thresholds
        if thresholds:
            initial_thresholds[1:-1] = [np.mean(t) for t in thresholds]

    # Apply the final thresholds
    result = np.zeros_like(gray_image)
    for i in range(num_levels - 1):
        result[
            (gray_image >= initial_thresholds[i]) & (gray_image <= initial_thresholds[i + 1])
        ] = 255 * (i + 1) // num_levels

    return result, initial_thresholds

# Allow the user to input the image path
image_path = input("Enter the path to your image: ")

# Load an image with error handling
input_image = cv2.imread(image_path)
if input_image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# Specify the number of levels for multilevel thresholding within the range of 1 to 255
num_levels = int(input("Enter the number of levels for multilevel thresholding (1 to 255): "))
num_levels = np.clip(num_levels, 1, 255)  # Ensure the value is within the valid range

# Perform multilevel thresholding and get thresholds
output_image, thresholds = multilevel_thresholding(input_image, num_levels)

# Plot the original image, multilevel thresholded image, and histogram in a single figure
plt.figure(figsize=(10, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Multilevel Thresholded Image
plt.subplot(2, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title("Multilevel Thresholded Image using OTUSU method")
plt.axis("off")

# Histogram for Multilevel Thresholded Image
plt.subplot(2, 2, 3)
plt.title("Histogram for Multilevel Thresholded Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.hist(output_image.flatten(), bins=num_levels, range=[0, 256], color='black', histtype='step')

plt.show()
