import cv2
import numpy as np

def threshold_image(frame, threshold_value, blur_value):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    _, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])


cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thresholded", 1280, 720)

cv2.createTrackbar("Threshold", "Thresholded", 0, 255, lambda x: None)
cv2.createTrackbar("Blur", "Thresholded", 1, 21, lambda x: None)

cap = cv2.VideoCapture("videos/demo.mp4")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    threshold_value = cv2.getTrackbarPos("Threshold", "Thresholded")
    blur_value = cv2.getTrackbarPos("Blur", "Thresholded")
    if blur_value % 2 == 0:
        blur_value+=1
    
    white_mask = cv2.inRange(frame, lower_white, upper_white)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=white_mask)

    thresholded = threshold_image(filtered_frame, threshold_value, blur_value)
    
    stacked = np.hstack((frame, cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Thresholded", stacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
