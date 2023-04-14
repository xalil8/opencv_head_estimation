import cv2
import numpy as np

# Define the polygon vertices
pts = np.array( [[1315, 627], [1912, 634], [1908, 582], [1340, 325], [736, 278]], np.int32)
pts2 = np.array( [[1287, 53], [1909, 232], [1909, 566], [1310, 299]], np.int32)

# Define the metal plate polygon vertices
metal_pts = np.array( [[1298, 624], [1324, 626], [1373, 337], [1345, 338]], np.int32)

# Create a mask from the polygons
mask = np.zeros((1080, 1920), np.uint8)
cv2.fillPoly(mask, [pts], 255)
cv2.fillPoly(mask, [pts2], 255)
cv2.fillPoly(mask, [metal_pts], 255)  # fill metal plate area with white

# Open the video stream
cap = cv2.VideoCapture('videos/demo.mp4')

# Initialize variables to keep track of shape presence
previous_count = 0
shape_present = False
threshold = 50000

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Apply white filter to the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Apply the polygon mask to the white filtered frame
    masked_frame = cv2.bitwise_and(mask_white, mask_white, mask=mask)

    # Find contours of white pixels in the masked frame
    contours, _ = cv2.findContours(masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the area of the largest white contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        count = cv2.contourArea(largest_contour)
    else:
        count = 0

    # If the area of the largest contour is larger than a threshold, consider it a shape
    if count > threshold:
        shape_present = True
        if not previous_count:
            print("Shape appeared!")
    else:
        shape_present = False
        if previous_count:
            print("Shape disappeared!")

    previous_count = shape_present

    # Draw the polygons on the frame
    cv2.polylines(frame, [pts], True, (0, 255, 0), thickness=2)
    cv2.polylines(frame, [pts2], True, (0, 255, 0), thickness=2)
    cv2.polylines(frame, [metal_pts], True, (0, 0, 255), thickness=2)  # draw metal plate polygon in red

    # Draw the largest white contour on a copy of the frame and show it in a separate window
    frame_copy = frame.copy()
    if len(contours) > 0:
        if count > threshold:
            cv2.drawContours(frame_copy, [largest_contour], 0, (0, 0, 255), thickness=2)
    # Convert the masked frame to a 3-channel image
    masked_frame_color = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)
    # Draw the largest white contour on a copy of the frame and show it in a separate window
    frame_copy = frame.copy()
    if len(contours) > 0:
        if count > threshold:
            cv2.drawContours(frame_copy, [largest_contour], 0, (0, 0, 255), thickness=2)
    # Convert the masked frame to a 3-channel image
    masked_frame_color = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)
    # Horizontally stack the masked and contour images
    stacked_image = np.hstack((masked_frame_color, frame_copy))
    cv2.imshow('Shape Contour', stacked_image)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()