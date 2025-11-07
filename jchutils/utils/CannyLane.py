import numpy as np
import cv2

def canny_edges(
    gray: np.ndarray,
    th1: int = 82,
    th2: int = 177,
    blur_ksize: int = 9
) -> np.ndarray:
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, threshold1=th1, threshold2=th2)
    
    return edges
