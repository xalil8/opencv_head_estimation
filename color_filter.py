import cv2
import numpy as np

# Define the lower and upper white color thresholds in BGR color space
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])

# Create a video capture object
cap = cv2.VideoCapture("videos/demo.mp4")

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        break

    # Apply the white color filter
    white_mask = cv2.inRange(frame, lower_white, upper_white)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=white_mask)

    # Display the original and filtered frames
    stacked_frame = np.hstack((frame, filtered_frame))
    
    cv2.imshow('Original vs Filtered', stacked_frame)

    # Adjust the white color thresholds using trackbars
    def nothing(x):
        pass

    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('Low B', 'Trackbars', lower_white[0], 255, nothing)
    cv2.createTrackbar('Low G', 'Trackbars', lower_white[1], 255, nothing)
    cv2.createTrackbar('Low R', 'Trackbars', lower_white[2], 255, nothing)
    cv2.createTrackbar('High B', 'Trackbars', upper_white[0], 255, nothing)
    cv2.createTrackbar('High G', 'Trackbars', upper_white[1], 255, nothing)
    cv2.createTrackbar('High R', 'Trackbars', upper_white[2], 255, nothing)

    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Get the current trackbar positions
    low_b = cv2.getTrackbarPos('Low B', 'Trackbars')
    low_g = cv2.getTrackbarPos('Low G', 'Trackbars')
    low_r = cv2.getTrackbarPos('Low R', 'Trackbars')
    high_b = cv2.getTrackbarPos('High B', 'Trackbars')
    high_g = cv2.getTrackbarPos('High G', 'Trackbars')
    high_r = cv2.getTrackbarPos('High R', 'Trackbars')

    # Update the lower and upper white color thresholds
    lower_white = np.array([low_b, low_g, low_r])
    upper_white = np.array([high_b, high_g, high_r])

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
