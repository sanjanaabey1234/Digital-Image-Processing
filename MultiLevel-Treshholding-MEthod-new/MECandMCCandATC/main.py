import cv2
import numpy as np
import matplotlib.pyplot as plt

# cv2: OpenCV library for computer vision tasks.
# numpy (np): Numerical operations library for efficient array manipulations.
# matplotlib.pyplot (plt): Library for plotting and visualization.

def multilevel_thresholding_mec_atc_mcc(image, num_levels):
    # multilevel_thresholding_mec_atc_mcc:
    # Function to perform multilevel thresholding using MEC (Minimum Error Criterion),
    # ATC (Adaptive Thresholding by Entropy with Conditional expectation),
    # and MCC (Minimum Conditional Entropy Criterion).

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate initial thresholds using MEC
    initial_thresholds = np.linspace(0, 255, num_levels + 2)[1:-1]

    # Refine thresholds using iterative MEC
    for _ in range(10):  # You can adjust the number of iterations
        thresholds = []
        for i in range(num_levels - 1):
            region = gray_image[
                (gray_image >= initial_thresholds[i]) & (gray_image <= initial_thresholds[i + 1])
            ]
            if region.size > 0:  # Check if the region is not empty
                probabilities, _ = np.histogram(region, bins=256, range=[0, 256], density=True)
                cumulative_sum = np.cumsum(probabilities)
                entropy_values = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                weighted_entropies = []
                for t in range(256):
                    if t <= initial_thresholds[i] or t >= initial_thresholds[i + 1]:
                        continue
                    p1 = cumulative_sum[int(initial_thresholds[i])]
                    p2 = cumulative_sum[int(t)] - cumulative_sum[int(initial_thresholds[i])]
                    p3 = cumulative_sum[int(initial_thresholds[i + 1])] - cumulative_sum[int(t)]
                    if p1 > 0 and p2 > 0 and p3 > 0:
                        entropy_value = -p1 * np.log2(p1 + 1e-10) - p2 * np.log2(p2 + 1e-10) - p3 * np.log2(p3 + 1e-10)
                        weighted_entropies.append(entropy_value)
                if weighted_entropies:
                    best_threshold = np.argmax(weighted_entropies) + int(initial_thresholds[i]) + 1
                    thresholds.append(best_threshold)

        # Update initial thresholds using MEC
        if thresholds:
            initial_thresholds[1:-1] = thresholds

    # Apply the final thresholds using ATC
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

# Perform multilevel thresholding using MEC, ATC, and MCC and get thresholds
output_image, thresholds = multilevel_thresholding_mec_atc_mcc(input_image, num_levels)

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
plt.title("Multilevel Thresholded Image (MEC, ATC, MCC)")
plt.axis("off")

# Histogram for Multilevel Thresholded Image
plt.subplot(2, 2, 3)
plt.title("Histogram for Multilevel Thresholded Image")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.hist(output_image.flatten(), bins=num_levels, range=[0, 256], color='black', histtype='step')

plt.show()
