import cv2
import numpy as np


def load_image(f: str) -> np.ndarray:
    """
    Load an image from a file.
    :param f: Path to the image file.
    :return: Image as a numpy array.
    """
    return cv2.imread(f, cv2.IMREAD_UNCHANGED)


def crop_center_power_of_2(image: np.ndarray) -> np.ndarray:
    """
    Crop the center of the image to the nearest power of 2 size.
    :param image: Input image as a numpy array.
    :return: Cropped image.
    """
    h, w = image.shape[:2]
    new_size = min(h, w)
    new_size = 2 ** int(np.log2(new_size))
    start_h = (h - new_size) // 2
    start_w = (w - new_size) // 2
    return image[start_h : start_h + new_size, start_w : start_w + new_size]


def average_2x2(image: np.ndarray) -> np.ndarray:
    """
    Average a 2x2 block of pixels in the image.
    :param image: Input image as a numpy array.
    :return: Image with averaged 2x2 blocks.
    """
    h, w = image.shape[:2]
    new_h, new_w = h // 2, w // 2
    avg_image = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            avg_image[i, j] = np.mean(
                image[i * 2 : i * 2 + 2, j * 2 : j * 2 + 2], axis=(0, 1)
            )

    return avg_image


def reproject_image(
    image: np.ndarray,
    focal_length_in_pixels: float,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    """
    Reproject the image based on the given camera parameters.
    :param image: Input image as a numpy array.
    :param focal_length_in_pixels: Focal length in pixels.
    :param a b c: Camera rotation vector.
    :return: Reprojected image.
    """
    h, w = image.shape[:2]
    K = np.array([[focal_length_in_pixels, 0, w / 2], [0, focal_length_in_pixels, h / 2], [0, 0, 1]])
    R = cv2.Rodrigues(np.array([a, b, c]))[0]
    RT = np.hstack((R, np.zeros((3, 1))))
    P = K @ RT

    reprojected_image = cv2.warpPerspective(image, P, (w, h), flags=cv2.INTER_LINEAR)
    return reprojected_image