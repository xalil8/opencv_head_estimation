import cv2
import numpy as np


cv2.namedWindow('STACKED OUTPUT',cv2.WINDOW_NORMAL)
cv2.resizeWindow('STACKED OUTPUT', 720, 1080)

# Define the polygon vertices
pts = np.array( [[1315, 627], [1912, 634], [1908, 582], [1340, 325], [736, 278]], np.int32)
pts2 = np.array( [[1287, 53], [1909, 232], [1909, 566], [1310, 299]], np.int32)

polygons = [
            [[1041, 469], [1068, 485], [1106, 301], [1071, 298]],
            [[808, 333], [839, 354], [848, 281], [821, 271]],
            [[1342, 315], [1384, 329], [1332, 649], [1281, 639]],
            [[1458, 354], [1497, 375], [1430, 647], [1392, 645]],
            [[1470, 281], [1515, 303], [1561, 101], [1512, 86]],
            [[1377, 50], [1349, 211], [1393, 255], [1426, 65]]]

# Create a mask from the polygon
mask = np.zeros((1080, 1920), np.uint8)

cv2.fillPoly(mask, [pts], 255)
cv2.fillPoly(mask, [pts2], 255)

# Open the video stream
cap = cv2.VideoCapture('videos/demo.mp4')

# Initialize variables to keep track of shape presence
previous_count = 0
shape_present = False
threshold = 50000
lower_white = np.array([22, 122, 179])
upper_white = np.array([255, 255, 255])
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break
    for polygon in polygons:
        cv2.fillPoly(frame, [np.array(polygon)], (255,255,255))
    # Apply white filter to the frame
    mask_white = cv2.inRange(frame, lower_white, upper_white)

    # Apply the polygon mask to the white filtered frame
    masked_frame = cv2.bitwise_and(mask_white, mask_white, mask=mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    masked_frame = cv2.morphologyEx(masked_frame, cv2.MORPH_OPEN, kernel)
    # Find contours of white pixels in the masked frame
    contours, _ = cv2.findContours(masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)

    # Calculate the area of the largest white contour
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        count = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)

    else:
        count = 0

    # If the area of the largest contour is larger than a threshold, consider it a shape
    if count > threshold:
        shape_present = True
        if not previous_count:
            print("ITEM PASSING LINE!")
        cv2.putText(masked_frame,str("ITEM PASSING LINE!"),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    else:
        shape_present = False
        if previous_count:
            print("NO ITEM ON LINE!")
        cv2.putText(masked_frame,str("NO ITEM ON LINE!"),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    previous_count = shape_present

    # Draw the polygon on the frame
    cv2.polylines(masked_frame, [pts], True, (233, 33, 0), thickness=2)
    cv2.polylines(masked_frame, [pts2], True, (233, 33, 0), thickness=2)

    # Draw the largest white contour on a copy of the frame and show it in a separate window
    frame_copy = frame.copy()
    if len(contours) > 0:
        if count > threshold:
            cv2.drawContours(masked_frame, [hull], 0, (0, 255, 0), thickness=2)
    # Convert the masked frame to a 3-channel image
    #masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)
    # Horizontally stack the masked and contour images
    #stacked_image = np.vstack((masked_frame_color, frame_copy))
    cv2.imshow('STACKED OUTPUT', masked_frame)
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
