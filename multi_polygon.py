import cv2
import numpy as np

# Open the input video file
input_file = cv2.VideoCapture('videos/demo.mp4')

# Get the video properties
fps = input_file.get(cv2.CAP_PROP_FPS)
width = int(input_file.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_file.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the coordinates of the polygons
polygons = [[[819, 331], [833, 338], [844, 282], [826, 282]],
            [[1052, 471], [1061, 476], [1095, 304], [1085, 304]],
            [[1344, 331], [1373, 339], [1324, 629], [1297, 625]],
            [[1462, 376], [1485, 387], [1426, 636], [1401, 637]]]

# Loop through the frames of the input video
while True:
    ret, frame = input_file.read()
    if not ret:
        break
    # Create a blank mask for this frame
    # Draw the polygons onto the mask
    for polygon in polygons:
        cv2.fillPoly(frame, [np.array(polygon)], (255,255,255))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)


    cv2.imshow('Result', thresh)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the input and output video files
input_file.release()
cv2.destroyAllWindows()
