import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

video_path = 'road_video.avi'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 720

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'H264')
out1 = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, frame_size)


# Color filter function
def apply_color_filter(image):
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower_white = np.array([20, 150, 20])
    upper_white = np.array([255, 255, 255])

    lower_yellow = np.array([0, 85, 81])
    upper_yellow = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(hls_image, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hls_image, lower_white, upper_white)

    mask = cv2.bitwise_or(yellow_mask, white_mask)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    return filtered_image

# Region of Interest function
def apply_roi(image):
    height, width = image.shape[0], image.shape[1]

    vertices = np.array([
        [int(0.1 * width), height],
        [int(0.1 * width), int(0.1 * height)],
        [int(0.4 * width), int(0.1 * height)],
        [int(0.4 * width), height],
        [int(0.7 * width), height],
        [int(0.7 * width), int(0.1 * height)],
        [int(0.9 * width), int(0.1 * height)],
        [int(0.9 * width), height],
        [int(0.2 * width), height]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# Perspective wrapping function
def apply_perspective_warp(image):
    height, width = image.shape[0], image.shape[1]

    src_points = np.float32([[width // 2 - 30, height * 0.53], [width // 2 + 60, height * 0.53], [width * 0.3, height],
                             [width, height]])
    dst_points = np.float32([[0, 0], [width - 350, 0], [400, height], [width - 150, height]])

    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    warped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    return warped_image, inverse_transform_matrix

# Histogram plotting function
def plot_histogram(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    return left_base, right_base

# Sliding window search function
def sliding_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    n_windows = 4
    window_height = np.int(binary_warped.shape[0] / n_windows)
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    color = [0, 255, 0]
    thickness = 2

    for window in range(n_windows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)

        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current = np.int(np.mean(nonzero_x[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_fitx = np.trunc(left_fitx)
    right_fitx = np.trunc(right_fitx)

    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    result = {'left_fitx': left_fitx, 'right_fitx': right_fitx, 'ploty': ploty}

    return result

# Draw lane lines on the original image
def draw_lane_lines(original_image, warped_image, inverse_transform_matrix, draw_info):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
    
    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    new_warp = cv2.warpPerspective(color_warp, inverse_transform_matrix, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, new_warp, 0.4, 0)

    return pts_mean, result

# Main loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply color filter
    filtered_frame = apply_color_filter(frame)

    # Apply region of interest
    roi_frame = apply_roi(filtered_frame)

    # Apply perspective warp
    warped_frame, inverse_perspective_matrix = apply_perspective_warp(roi_frame)

    # Histogram plot
    left_base, right_base = plot_histogram(warped_frame)

    # Sliding window search
    search_result = sliding_window_search(warped_frame, left_base, right_base)

    # Draw lane lines on the original frame
    lane_info, result_frame = draw_lane_lines(frame, warped_frame, inverse_perspective_matrix, search_result)

    # Write the result frame to the output video
    out_video.write(result_frame)

# Release video capture and writer
cap.release()
out_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()