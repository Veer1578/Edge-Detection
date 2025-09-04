
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(title, image):
    '''Utility function to display an image'''
    plt.figure(figsize=(8, 8))
    if len(image.shape) == 2:   # Grayscale
        plt.imshow(image, cmap='gray')
    else:   # Convert image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def interactive_edge_detection(image_path):
    '''Interactive activity for edge detection and filtering'''
    image_path = input('Enter image path to your image:')
    image = cv2.imread(image_path)

    if image is None:
        print('Error. Image not found')
        return

    # Convert to gray image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Orignal Grayscaled Image {gray_image}")

    print("Select an option")
    print("1. Sobel Edge Detection")
    print("2. Canny Edge Detection")
    print("3. Laplacian Edge Detection")
    print("4. Gaussian Smoothing")
    print("5. Median Filtering")
    print("6. Exit")

    while True:
        choice = input('Enter choice (1-6): ')

        if choice == '1':
            # Sobel Edge Detection
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            combined_sobel = cv2.bitwise_or(
                sobelx.astype(np.uint8), sobely.astype(np.uint8))
            display_image('Sobel Edge Detection', combined_sobel)
        elif choice == '2':
            # Canny Edge Detection
            print('Adjust thresholds for Canny (default: 100 and 200)')
            lower = int(input('Enter lower threshold: '))
            upper = int(input('Enter upper threshold: '))
            edges = cv2.Canny(gray_image, lower, upper)
            display_image('Canny Edge Detection', edges)
        elif choice == '3':
            # Laplacian Edge Detection
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            display_image('Laplacian Edge Detection',
                          np.abs(laplacian).astype(np.uint8))
        elif choice == '4':
            # Gaussian Smoothing
            print("Adjust kernel size for Gaussian blur (must be odd, default:5)")
            k_size = int(input('Kernel Size: '))
            blurred = cv2.GaussianBlur(gray_image, (k_size, k_size), 0)
            display_image('Gaussian Smoothing', blurred)
        elif choice == '5':
            # Median Filtering
            print("Adjust kernel size for Median filtering (must be odd, default:5)")
            k_size = int(input('Kernel Size: '))
            median_fileterd = cv2.medianBlur(gray_image, k_size)
            display_image('Median Filtering', median_fileterd)
        else:
            print('Exiting...')
            break


# Provide path to image
interactive_edge_detection('image for project.jpg')
