import math
import numpy as np
import cv2

cam2lidar_len = 6.8 # cm
cam_h = 10.3 # cm
cam_image_zero = 18 # cm

cam_angle_view_v = 120 # degree

const_ld_cm = 0.1 # 1 ld = 0.1 cm

cam2lidar_len_ld = cam2lidar_len / const_ld_cm # ld
cam_h_ld =  cam_h / const_ld_cm # ld
cam_image_zero_ld = cam_image_zero / const_ld_cm # ld

line_range = 400 # ld
line_width = 400 # ld

lidar_max_range = 400 # ld

rslt_img_size = 400 # px

scale = rslt_img_size / (lidar_max_range * 2)

lidar_color = (0, 255, 0)
lidar_jajucha_color = (0, 0, 255)
line_color = (255, 0, 0)

# simple math
def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

# draw imgs
def draw_lidar(theta, dist):
    theta_rad = np.deg2rad(-theta)

    x = dist * np.cos(theta_rad)
    y = dist * np.sin(theta_rad)
    
    # max_dist = 5000

    ix = (x * scale + rslt_img_size / 2).astype(np.int32)
    iy = (y * scale + rslt_img_size / 2).astype(np.int32)

    img = np.zeros((rslt_img_size, rslt_img_size, 3), dtype=np.uint8)

    for nx, ny in zip(ix, iy):
        if (ix >= 0) and (ix < rslt_img_size) and (iy >= 0) and (iy < rslt_img_size):
            cv2.circle(img, (nx, ny), radius=2, color=lidar_color, thickness=-1)

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img

def draw_canny(img):
    # img : numpy array - cannyed image
    canny_w = img.shape[1]
    canny_h = img.shape[0]

    srcroi_h = int(canny_h * 2 / (line_range - cam2lidar_len_ld) * (line_range - cam_image_zero_ld - cam2lidar_len_ld))
    canny_w_ld = cam_image_zero_ld * math.tan(math.radians(cam_angle_view_v / 2)) * 2
    srcroi_w_half = int(canny_w * (line_width / canny_w_ld) / 2)

    src_points = np.float32([
        [canny_w // 2 - srcroi_w_half * (canny_h - srcroi_h * 2) / canny_h, canny_h - srcroi_h], 
        [canny_w // 2 + srcroi_w_half * (canny_h - srcroi_h * 2) / canny_h, canny_h - srcroi_h], 
        [canny_w // 2 + srcroi_w_half, canny_h], 
        [canny_w // 2 - srcroi_w_half, canny_h] 
    ]) 

    dst_w = line_width * scale
    dst_h = (line_range - cam_image_zero_ld - cam2lidar_len_ld) * scale

    dst_points = np.float32([
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
        [0, dst_h]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(img, matrix, (dst_w, dst_h))

    # change color to line_color
    warped_img_colored = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)
    warped_img_colored[np.where((warped_img_colored != [0, 0, 0]).all(axis=2))] = line_color

    return warped_img

def draw_map(img, theta, d):
    lidar_img = draw_lidar(theta, d)
    cannyed_img = draw_canny(img)

    # 수정 확인 - 위치
    roi = lidar_img[0:cannyed_img.shape[0], rslt_img_size//2 - cannyed_img.shape[1]//2:rslt_img_size//2 + cannyed_img.shape[1]//2]
    roi[np.where((cannyed_img != [0, 0, 0]).all(axis=2))] = cannyed_img[np.where((cannyed_img != [0, 0, 0]).all(axis=2))]
    # roi = cv2.addWeighted(roi, 1, cannyed_img, 1, 0)

    rslt_img = lidar_img.copy()
    rslt_img[0:roi.shape[0], rslt_img.shape[1]//2 - roi.shape[1]//2:rslt_img.shape[1]//2 + roi.shape[1]//2] = roi

    return rslt_img