import cv2
import numpy as np

def threshold_image(frame, threshold_value, blur_value):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    _, thresholded = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thresholded", 1280, 720)

cv2.createTrackbar("Threshold", "Thresholded", 0, 255, lambda x: None)
cv2.createTrackbar("Blur", "Thresholded", 1, 21, lambda x: None)

cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cropped", 640, 360)

cap = cv2.VideoCapture("demo.mp4")
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    threshold_value = cv2.getTrackbarPos("Threshold", "Thresholded")
    blur_value = cv2.getTrackbarPos("Blur", "Thresholded")
    if blur_value % 2 == 0:
        blur_value+=1
    thresholded = threshold_image(frame, threshold_value, blur_value)
    
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest_contour)
    cropped = frame[y:y+h, x:x+w]
    
    stacked = np.hstack((frame, cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Thresholded", stacked)
    cv2.imshow("Cropped", cropped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
