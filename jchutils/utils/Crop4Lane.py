import numpy as np
import cv2

#       ------------------------------------------------
#       |                                              |
#       |                                              |
#       |      wl                              wr      |
#. . . .|. . . . . . . .---------------- . . . . . . . .
#       |            /                    \            |
#       |        /                            \        |     res[1]
#       |    /                                    \    |
#    H  |/                                            \|
#       |                                              |
#       |  h                                           |
#       |                                              |
#       ------------------------------------------------
#                           res[0]
# 
# 아스키아트 손수 그린건데 진짜 겁나 힘들었다...

def crop_roi(
    img: np.ndarray, 
    res: tuple[int, int] = (640, 400),
    wl: int = 180,
    wr: int = 180,
    h: int = 64,
    H: int = 120,
) -> np.ndarray: 
    crop_points_L = np.float32([
        [0, 0],
        [wl, 0],
        [0, h]
    ])
    crop_points_R = np.float32([
        [res[0], 0],
        [res[0] - wr, 0],
        [res[0], h]
    ])

    croped_img = img[res[1] - H:res[1], :]

    cv2.fillPoly(croped_img, np.int32([crop_points_L]), (0, 0, 0))
    cv2.fillPoly(croped_img, np.int32([crop_points_R]), (0, 0, 0))

    return croped_img
