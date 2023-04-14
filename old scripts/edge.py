import cv2
import numpy as np

def threshold_image(frame, threshold_value, blur_value):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    _, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    return thresholded, canny

cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thresholded", 1280, 720)



###########################TRACKBAR #########################################
cv2.createTrackbar("Threshold", "Thresholded", 0, 255, lambda x: None)
cv2.createTrackbar("Blur", "Thresholded", 1, 21, lambda x: None)
cv2.createTrackbar("Canny threshold 1", "Thresholded", 0, 255, lambda x: None)
cv2.createTrackbar("Canny threshold 2", "Thresholded", 0, 255, lambda x: None)
###########################TRACKBAR #########################################

cap = cv2.VideoCapture("demo.mp4")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    threshold_value = cv2.getTrackbarPos("Threshold", "Thresholded")
    blur_value = cv2.getTrackbarPos("Blur", "Thresholded")
    canny_threshold1 = cv2.getTrackbarPos("Canny threshold 1", "Thresholded")
    canny_threshold2 = cv2.getTrackbarPos("Canny threshold 2", "Thresholded")
    if blur_value % 2 == 0:
        blur_value+=1
    thresholded, canny = threshold_image(frame, threshold_value, blur_value)
    
    stacked = np.hstack((frame, cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR), cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Thresholded", stacked)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
