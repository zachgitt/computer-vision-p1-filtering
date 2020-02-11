import sys
import cv2
import numpy as np
from math import floor, exp


def inbounds(i, j, height, width):
    if 0 <= i < height and 0 <= j < width:
        return True
    return False


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # Save image dimensions
    height = img.shape[0]
    width = img.shape[1]
    depth = 1
    if len(img.shape) == 3:
        depth = img.shape[2]

    # Update image to 3d
    img = img.reshape((height, width, depth))

    # Save kernel dimensions
    m = kernel.shape[0]
    n = kernel.shape[1]

    # Create a new image    
    d3_img = np.zeros((height, width, depth))

    # Iterate image
    for i in range(height):
        for j in range(width):

            # Determine reference to index image
            i_start = i - int(floor(m / 2))
            j_start = j - int(floor(n / 2))

            # Kernelize depth of pixel
            for k in range(depth):

                # Kernelize each pixel
                sum = 0
                for i_kernel in range(m):
                    for j_kernel in range(n):

                        # Calculate img indices
                        i_img = i_start + i_kernel
                        j_img = j_start + j_kernel
                        if inbounds(i_img, j_img, height, width):
                            sum += img[i_img][j_img][k] * kernel[i_kernel][j_kernel]

                # Save kernelized image
                d3_img[i][j][k] = sum

    # Reshape grayscale
    new_img = d3_img
    if depth == 1:
        new_img = d3_img.reshape((height, width))

        # Return transformed image
    return new_img


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img, np.flip(kernel))


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    kernel = np.zeros((height, width))
    center_i = int(floor(height / 2))
    center_j = int(floor(width / 2))
    for i in range(height):
        for j in range(width):
            x = (center_i - i)**2
            y = (center_j - j)**2
            kernel[i][j] = exp(-(x + y)/(2 * sigma**2))

    # Normalize
    norm = np.sum(kernel)
    kernel = np.true_divide(kernel, norm)

    return kernel


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter suppresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return np.subtract(img, low_pass(img, sigma, size))


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
