import torch
import cv2
import numpy as np
import time 
from ultralytics import YOLO
import math


class PoseEstimation:
    def __init__(self, model_path='models/yolov8n-pose.pt',source_video_path = "videos/demo.mp4"):
        self.model = YOLO(model_path)
        self.video_write = False
        self.shape_present = False
        self.threshold = 50000
        self.previous_count = 0
        self.frame_counter = 0
        self.count = 0
        self.paused = False
        video_saving_path = source_video_path[:len(source_video_path)-4:]+"_output.mp4"
        self.video_cap = cv2.VideoCapture(source_video_path)
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        width, height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        desired_fps = 35
        if self.video_write:
            self.video = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,fps, (width,height))

        cv2.namedWindow('STACKED OUTPUT',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("STACKED OUTPUT",(720,720))

        self.lower_white = np.array([22, 122, 179])
        self.upper_white = np.array([255, 255, 255])
        # Define the polygon vertices
        self.pts = np.array( [[1315, 627], [1912, 634], [1908, 582], [1340, 325], [736, 278]], np.int32)
        self.pts2 = np.array( [[1287, 53], [1909, 232], [1909, 566], [1310, 299]], np.int32)
        # Create a mask from the polygon
        self.polygons =  [[[1041, 469], [1068, 485], [1106, 301], [1071, 298]],
                    [[808, 333], [839, 354], [848, 281], [821, 271]],
                    [[1342, 315], [1384, 329], [1332, 649], [1281, 639]],
                    [[1458, 354], [1497, 375], [1430, 647], [1392, 645]],
                    [[1470, 281], [1515, 303], [1561, 101], [1512, 86]],
                    [[1377, 50], [1349, 211], [1393, 255], [1426, 65]]]

        self.mask = np.zeros((1080, 1920), np.uint8)
        cv2.fillPoly(self.mask, [self.pts], 255)
        cv2.fillPoly(self.mask, [self.pts2], 255)

    def find_distance(self,A,B):
        x1,y1 = A[0],A[1]
        x2,y2 = B[0],B[1]
        # Calculate the distance between two points using the distance formula
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return int(distance) 

    def calculate_angle(self,A,B,C):
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
    
    def keypoint_model(self, frame):
        results = self.model.predict(source=frame,device="cpu")
      
        for result in results:
            keypoints = result[0].keypoints

            # Define the desired keypoints#############################################################
            nose = keypoints[0][:2]
            eye_left = keypoints[1][:2]
            eye_right = keypoints[2][:2]
            ear_left = keypoints[3][:2]
            ear_right = keypoints[4][:2]



            shoulder_left = keypoints[5][:2]
            shoulder_right = keypoints[6][:2]
            elbow_left = keypoints[7][:2]
            elbow_right = keypoints[8][:2]

            # Convert keypoints to integers#############################################################
            nose = (int(nose[0]), int(nose[1]))
            eye_left = (int(eye_left[0]), int(eye_left[1]))
            eye_right = (int(eye_right[0]), int(eye_right[1]))
            ear_left = (int(ear_left[0]), int(ear_left[1]))
            ear_right = (int(ear_right[0]), int(ear_right[1]))



            shoulder_left = (int(shoulder_left[0]), int(shoulder_left[1]))
            shoulder_right = (int(shoulder_right[0]), int(shoulder_right[1]))
            elbow_left = (int(elbow_left[0]), int(elbow_left[1]))
            elbow_right = (int(elbow_right[0]), int(elbow_right[1]))

            ###################DRAW POINTS ###########################################
            cv2.circle(frame, (nose[0],nose[1]), 1, (30,133,233), 10)
            cv2.circle(frame, (eye_left[0],eye_left[1]), 1, (30,133,233), 10)
            cv2.circle(frame, (eye_right[0],eye_right[1]), 1, (30,133,233), 10)
            cv2.circle(frame, (ear_left[0],ear_left[1]), 1, (30,133,233), 10)
            cv2.circle(frame, (ear_right[0],ear_right[1]), 1, (30,133,233), 10)

            cv2.circle(frame, (shoulder_left[0],shoulder_left[1]), 1, (30,133,233), 10)
            cv2.circle(frame, (shoulder_right[0],shoulder_right[1]), 1, (30,133,233), 10)
            cv2.circle(frame, (elbow_right[0],elbow_right[1]), 1, (30,133,233), 10)
            cv2.circle(frame, (elbow_left[0],elbow_left[1]), 1, (30,133,233), 10)

            ####################### DRAW LINES BETWEEN KEYPOINTS ###############################
            #cv2.line(frame, ear_left, shoulder_left, (233,233,10), thickness=2)
            #cv2.line(frame, ear_right, shoulder_right, (233,233,10), thickness=2)
            cv2.line(frame, ear_right, ear_left, (233,233,10), thickness=2)

            cv2.line(frame, eye_left, eye_right, (233,233,10), thickness=2)
            #cv2.line(frame, shoulder_left, elbow_left, (233,233,10), thickness=2)



            cv2.line(frame, shoulder_left, elbow_left, (233,233,10), thickness=2)
            cv2.line(frame, shoulder_right, elbow_right, (233,233,10), thickness=2)
            cv2.line(frame, shoulder_right, shoulder_left, (233,233,10), thickness=2)

            # ######################### ANGLES CALCULATION ####################################
            left_shoulder_angle = 180 - self.calculate_angle(A=elbow_left, B=shoulder_left, C=shoulder_right)
            right_shoulder_angle = 180 - self.calculate_angle(A=shoulder_left, B=shoulder_right, C=elbow_right)

            ########################   PUT ANGLES ON FRAME #####################################
            #cv2.putText(frame,str(left_shoulder_angle),(shoulder_left[0]-40,shoulder_left[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            #cv2.putText(frame,str(right_shoulder_angle),(shoulder_right[0]-40,shoulder_right[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            ########################    DISTANCE CALCULATION #####################################
            dis_ear = self.find_distance(ear_left,ear_right)
            dis_eyes = self.find_distance(eye_left,eye_right)
            ########################   PUT DISTANCE  ON FRAME #####################################
            cv2.putText(frame,str(dis_ear),(int((ear_left[0]+ear_right[0])/2),int((ear_right[1]+ear_left[1])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,str(dis_eyes),(int((eye_left[0]+eye_right[0])/2),int((eye_right[1]+eye_left[1])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            #if ear_right[0]< ear_left[0]:
            if ear_left[0]<550:
                if dis_ear > 70:
                    cv2.putText(frame,str("GERI DONUS TESPIT EDILDI"),(20,120),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),6)
            else:
                if (eye_right[0]+eye_left[0])/2 < ear_right[0]:
                    cv2.putText(frame,str("GERI DONUS TESPIT EDILDI"),(20,120),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),6)


            #if left_shoulder_angle >110 and right_shoulder_angle >120:
            #    cv2.putText(frame,str("GERI DONUS TESPIT EDILDI"),(20,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),6)
            return frame 
    
    def main(self):

        
        while True:
            # Read a frame from the video stream
            ret, frame = self.video_cap.read()
            if not ret:
                break
            
            self.frame_counter += 1
            #ADJUST FPS
            if self.frame_counter % 1 != 0:
                continue

            for polygon in self.polygons:
                cv2.fillPoly(frame, [np.array(polygon)], (255,255,255))

            # Apply white filter to the frame
            mask_white = cv2.inRange(frame, self.lower_white, self.upper_white)

            # Apply the polygon mask to the white filtered frame
            masked_frame = cv2.bitwise_and(mask_white, mask_white, mask=self.mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
            masked_frame = cv2.morphologyEx(masked_frame, cv2.MORPH_OPEN, kernel)
            # Find contours of white pixels in the masked frame
            contours, _ = cv2.findContours(masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            del kernel,masked_frame,mask_white
            # Calculate the area of the largest white contour
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                self.count = cv2.contourArea(largest_contour)
                hull = cv2.convexHull(largest_contour)

            else:
                self.count = 0

            # If the area of the largest contour is larger than a threshold, consider it a shape
            if self.count > self.threshold:
                self.shape_present = True
                if not self.previous_count:
                    print("ITEM PASSING LINE!")
                cv2.putText(frame,str("URETIM HATTINDA URUN VAR"),(20,160),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                frame = self.keypoint_model(frame)
                
            else:
                self.shape_present = False
                if self.previous_count:
                    print("NO ITEM ON LINE!")
                cv2.putText(frame,str("URETIM HATTTINDA URUN YOK!"),(20,160),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

            self.previous_count = self.shape_present

            # Draw the polygon on the frame
            cv2.polylines(frame, [self.pts], True, (0, 255, 0), thickness=2)
            cv2.polylines(frame, [self.pts2], True, (0, 255, 0), thickness=2)

            # Draw the largest white contour on a copy of the frame and show it in a separate window
            if len(contours) > 0:
                if self.count > self.threshold:
                    cv2.drawContours(frame, [hull], 0, (0, 0, 255), thickness=2)

            if not self.video_write:
                cv2.imshow('STACKED OUTPUT', frame)
                #cv2.imshow("ROI",frame)
            if self.video_write:
                print(f"frame {self.frame_counter} writing")
                self.video.write(frame)

            if self.paused:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(60)

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused

        self.video_cap.release()
        if self.video_write:
            self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("code has started")
    pose = PoseEstimation('models/yolov8n-pose.pt')
    pose.main()
    print("done")