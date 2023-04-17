import cv2
import numpy as np

# Define the lower and upper color thresholds in BGR color space
lower_color = np.array([22, 122, 179])
upper_color = np.array([255, 255, 255])

# Create a video capture object
cap = cv2.VideoCapture("videos/demo.mp4")

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Apply the color filter
    color_mask = cv2.inRange(frame, lower_color, upper_color)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
    _,thresholded = cv2.threshold(filtered_frame,40,255,cv2.THRESH_BINARY)
    # Display the original and filtered frames
    
    #eroded = cv2.erode(thresholded, None, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    stacked_frame = np.hstack((opening, thresholded))
    
    cv2.imshow('Original vs Filtered', stacked_frame)

    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
