import numpy as np
import cv2

def canny_edges(
    gray: np.ndarray,
    th1: int = 21,
    th2: int = 51,
    blur_ksize: int = 17
) -> np.ndarray:
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, threshold1=th1, threshold2=th2, L2gradient=1)
    
    return edges
