import torch
import cv2
import numpy as np
import time 
from ultralytics import YOLO
import math
global video_write
global model 
model = YOLO('models/yolov8n-pose.pt')  # load an official model

video_write = False

def find_distance(A,B):
    x1,y1 = A[0],A[1]
    x2,y2 = B[0],B[1]
    # Calculate the distance between two points using the distance formula
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return int(distance) 

def calculate_angle(A,B,C):
    # Calculate the vectors AB and BC
    AB = (B[0] - A[0], B[1] - A[1])
    BC = (C[0] - B[0], C[1] - B[1])

    # Calculate the dot product of the vectors AB and BC
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]

    # Calculate the magnitudes of the vectors AB and BC
    magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)

    # Calculate the angle between the vectors AB and BC
    angle = math.acos(dot_product / (magnitude_AB * magnitude_BC))

    # Convert the angle from radians to degrees
    angle_degrees = math.degrees(angle)
    return int(angle_degrees)

def keypoint_model(frame):
    results = model.predict(source=frame,device="cpu")
    for result in results:
        keypoints = result[0].keypoints
        confs = [int(keypoint[2] * 100) for keypoint in keypoints]
                
        nose = tuple(map(int, keypoints[0][:2]))
        eye_left = tuple(map(int, keypoints[1][:2]))
        eye_right = tuple(map(int, keypoints[2][:2]))
        ear_left = tuple(map(int, keypoints[3][:2]))
        ear_right = tuple(map(int, keypoints[4][:2]))
        shoulder_left = tuple(map(int, keypoints[5][:2]))
        shoulder_right = tuple(map(int, keypoints[6][:2]))
        elbow_left = tuple(map(int, keypoints[7][:2]))
        elbow_right = tuple(map(int, keypoints[8][:2]))
        wrist_left = tuple(map(int, keypoints[9][:2]))
        wrist_right = tuple(map(int, keypoints[10][:2]))
        hip_left = tuple(map(int, keypoints[11][:2]))
        hip_right = tuple(map(int, keypoints[12][:2]))
        knee_left = tuple(map(int, keypoints[13][:2]))
        knee_right = tuple(map(int, keypoints[14][:2]))
        ankle_left = tuple(map(int, keypoints[15][:2]))
        ankle_right = tuple(map(int, keypoints[16][:2]))

        keypoint_list = [nose, eye_left, eye_right, ear_left, ear_right,
                        shoulder_left, shoulder_right, elbow_left, elbow_right,
                        wrist_left, wrist_right, hip_left, hip_right,
                        knee_left, knee_right, ankle_left, ankle_right]
        
        ###################DRAW POINTS #####################################################
        for keypoint in keypoint_list:
            cv2.circle(frame, (keypoint[0],keypoint[1]), 1, (30,133,233), 10)

        ####################################################################################
        ####################### DRAW LINES BETWEEN KEYPOINTS ###############################

        #cv2.line(frame, ear_left, shoulder_left, (233,233,10), thickness=2)
        #cv2.line(frame, ear_right, shoulder_right, (233,233,10), thickness=2)
        cv2.line(frame, ear_right, ear_left, (233,233,10), thickness=2)
        cv2.line(frame, eye_left, eye_right, (233,233,10), thickness=2)
        #cv2.line(frame, shoulder_left, elbow_left, (233,233,10), thickness=2)
        cv2.line(frame, shoulder_left, elbow_left, (233,233,10), thickness=2)
        cv2.line(frame, shoulder_right, elbow_right, (233,233,10), thickness=2)
        cv2.line(frame, shoulder_right, shoulder_left, (233,233,10), thickness=2)
        
        ####################################################################################
        ############################## ANGLES CALCULATION ###################################
        left_shoulder_angle = 180 - calculate_angle(A=elbow_left, B=shoulder_left, C=shoulder_right)
        right_shoulder_angle = 180 - calculate_angle(A=shoulder_left, B=shoulder_right, C=elbow_right)


        ####################################################################################
        ########################   PUT ANGLES ON FRAME #####################################

        #cv2.putText(frame,str(left_shoulder_angle),(shoulder_left[0]-40,shoulder_left[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        #cv2.putText(frame,str(right_shoulder_angle),(shoulder_right[0]-40,shoulder_right[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        ####################################################################################
        ########################    DISTANCE CALCULATION ###################################
        dis_ear = find_distance(ear_left,ear_right)
        dis_eyes = find_distance(eye_left,eye_right)


        ####################################################################################
        ########################   PUT DISTANCE  ON FRAME ###################################
        cv2.putText(frame,str(dis_ear),(int((ear_left[0]+ear_right[0])/2),int((ear_right[1]+ear_left[1])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame,str(dis_eyes),(int((eye_left[0]+eye_right[0])/2),int((eye_right[1]+eye_left[1])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        ####################################################################################
        ####################################################################################


        #if ear_right[0]< ear_left[0]:
        if ear_left[0]<550:
            if dis_ear > 70:
                cv2.putText(frame,str("GERI DONUS TESPIT EDILDI"),(20,120),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),6)
        else:
            if (eye_right[0]+eye_left[0])/2 < ear_right[0]:
                cv2.putText(frame,str("GERI DONUS TESPIT EDILDI"),(20,120),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),6)
                
        return frame   

def main():
    eval = model("videos/image.png")
    del eval
    cv2.namedWindow('STACKED OUTPUT',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("STACKED OUTPUT",(720,720))
    # Define the polygon vertices
    pts = np.array( [[1315, 627], [1912, 634], [1908, 582], [1340, 325], [736, 278]], np.int32)
    pts2 = np.array( [[1287, 53], [1909, 232], [1909, 566], [1310, 299]], np.int32)
    shape_present = False
    threshold = 50000
    previous_count = 0
    frame_counter = 0
    count = 0
    paused = False
    lower_white = np.array([22, 122, 179])
    upper_white = np.array([255, 255, 255])

    polygons =  [[[1041, 469], [1068, 485], [1106, 301], [1071, 298]],
                [[808, 333], [839, 354], [848, 281], [821, 271]],
                [[1342, 315], [1384, 329], [1332, 649], [1281, 639]],
                [[1458, 354], [1497, 375], [1430, 647], [1392, 645]],
                [[1470, 281], [1515, 303], [1561, 101], [1512, 86]],
                [[1377, 50], [1349, 211], [1393, 255], [1426, 65]]]

    # Create a mask from the polygon
    mask = np.zeros((1080, 1920), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    cv2.fillPoly(mask, [pts2], 255)

    #####################################YOLO STUFF#######################################

    source_video_path = "videos/demo.mp4"
    video_saving_path = source_video_path[:len(source_video_path)-4:]+"_output.mp4"

    video_cap = cv2.VideoCapture(source_video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    desired_fps = 35

    if video_write:
        video = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,fps, (width,height))
    #####################################YOLO STUFF#######################################

    while True:
        # Read a frame from the video stream
        ret, frame = video_cap.read()
        if not ret:
            break
        
        frame_counter += 1

        #ADJUST FPS
        if frame_counter % 1 != 0:
            continue

        for polygon in polygons:
            cv2.fillPoly(frame, [np.array(polygon)], (255,255,255))
        # Apply white filter to the frame
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_white = cv2.inRange(frame, lower_white, upper_white)

        # Apply the polygon mask to the white filtered frame
        masked_frame = cv2.bitwise_and(mask_white, mask_white, mask=mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        masked_frame = cv2.morphologyEx(masked_frame, cv2.MORPH_OPEN, kernel)
        # Find contours of white pixels in the masked frame
        contours, _ = cv2.findContours(masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            cv2.putText(frame,str("URETIM HATTINDA URUN VAR"),(20,160),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            frame = keypoint_model(frame)
            
        else:
            shape_present = False
            if previous_count:
                print("NO ITEM ON LINE!")
            cv2.putText(frame,str("URETIM HATTTINDA URUN YOK!"),(20,160),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        previous_count = shape_present

        # Draw the polygon on the frame
        cv2.polylines(frame, [pts], True, (0, 255, 0), thickness=2)
        cv2.polylines(frame, [pts2], True, (0, 255, 0), thickness=2)

        # Draw the largest white contour on a copy of the frame and show it in a separate window
        #frame_copy = frame.copy()
        if len(contours) > 0:
            if count > threshold:
                cv2.drawContours(frame, [hull], 0, (0, 0, 255), thickness=2)
        # Convert the masked frame to a 3-channel image
        #masked_frame_color = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR)
        # Horizontally stack the masked and contour images
        #stacked_image = np.hstack((masked_frame_color, frame_copy))

        if not video_write:
            cv2.imshow('STACKED OUTPUT', frame)
            #cv2.imshow("ROI",frame)
        if video_write:
            print(f"frame {frame_counter} writing")
            video.write(frame)

        if paused:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(60)

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    video_cap.release()
    if video_write:
        video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("code has started")
    main()
    print("done")