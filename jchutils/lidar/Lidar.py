from typing import Callable
import numpy as np
import cv2

try:
    import jchm
except ModuleNotFoundError:
    pass

class Lidar:
    def __init__(self):
        try:
            get_lidar_data: Callable[[], tuple[np.ndarray, np.ndarray]] = jchm.lidar.get_lidar
        except NameError as e:
            raise ImportError("jchm 모듈을 사용할 수 없습니다.") from e
        self.get_lidar_data = get_lidar_data

    def getLidarData(self):
        return self.get_lidar_data()

    def getLidarImage(
        self, 
        show_jajucha: bool, 
        theta_array: np.ndarray, 
        dist_array: np.ndarray
    ) -> np.ndarray:
        # 자주차 jchm 코드에서 그대로 발췌

        theta_rad = np.deg2rad(-theta_array)

        # Compute Cartesian coordinates
        x = dist_array * np.cos(theta_rad)
        y = dist_array * np.sin(theta_rad)

        # Since OpenCV's origin is at the top-left corner, and y increases downwards,
        # we need to adjust the coordinates. Also, we need to scale the coordinates
        # to fit within the image size.

        # Define image size
        img_size = 400  # Adjust the image size as needed
        # max_dist = dist_array.max()
        # max_dist= 12000

        # Avoid division by zero if max_dist is zero
        max_dist = 5000
        if max_dist == 0:
            max_dist = 1

        # Scale factor to fit the points within the image
        scale = (img_size / 2) / max_dist

        # Shift and scale the coordinates to the center of the image
        x_img = (x * scale + img_size / 2).astype(np.int32)
        # y_img = (y * scale + img_size / 2).astype(np.int32)
        y_img = ((-y) * scale + img_size / 2).astype(np.int32)


        # Create a blank image
        image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        if show_jajucha:
            # Draw center position (LiDAR location) as a red dot
            # cv2.circle(image, (img_size // 2, img_size // 2), radius=4, color=(0, 0, 255), thickness=-1)

            # Draw center position (LiDAR location) as a red triangle
            triangle_radius = 10  # 삼각형 크기 조절
            center_x, center_y = img_size // 2, img_size // 2

            # 삼각형의 꼭짓점 세 개 (오른쪽을 향하는 정삼각형)
            triangle_pts = np.array([[
                (center_x + triangle_radius, center_y),  # 오른쪽
                (center_x - triangle_radius, center_y - triangle_radius),  # 왼쪽 위
                (center_x - triangle_radius, center_y + triangle_radius)   # 왼쪽 아래
            ]], dtype=np.int32)

            cv2.fillPoly(image, triangle_pts, color=(0, 0, 255))  # 빨간색 삼각형

        # Ensure the coordinates are within image boundaries
        valid_indices = (x_img >= 0) & (x_img < img_size) & (y_img >= 0) & (y_img < img_size)
        x_img = x_img[valid_indices]
        y_img = y_img[valid_indices]

        # Draw the points
        for xi, yi in zip(x_img, y_img):
            cv2.circle(image, (xi, yi), radius=2, color=(0, 255, 0), thickness=-1)

        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image
