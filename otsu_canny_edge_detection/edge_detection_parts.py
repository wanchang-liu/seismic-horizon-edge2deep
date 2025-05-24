import cv2
import numpy as np


def roberts(image):
    kernel_x = np.array([[-1, 0], [0, 1]], dtype=np.float32)
    kernel_y = np.array([[0, -1], [1, 0]], dtype=np.float32)

    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    edge_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edge_image


def prewitt(image):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    edge_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edge_image


def sobel(image, ksize=3):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    edge_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edge_image


def scharr(image):
    grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    edge_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return edge_image


def laplacian(image, ksize=3):
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    edge_image = cv2.convertScaleAbs(laplacian)
    return edge_image


# Canny
def canny(image, low_threshold, high_threshold):
    return cv2.Canny(image=image, threshold1=low_threshold, threshold2=high_threshold, L2gradient=True)


# Auto Canny
def auto_canny(image, sigma=0.33):
    # Compute the median of the single-channel pixel intensities
    v = np.median(image)

    # Use the computed median for automatic Canny edge detection
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(image=image, threshold1=lower, threshold2=upper, L2gradient=True)
    return edged


# Otsu_Canny
def otsu_canny(image, lowrate=0.5):
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    return cv2.Canny(image=image, threshold1=ret * lowrate, threshold2=ret, L2gradient=True)
