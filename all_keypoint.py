    #TODO make polygons filled with transparant color

import torch
import cv2
import numpy as np
import time 
from ultralytics import YOLO
import math
global video_write
video_write = True

def calculate_angle1(A, B, C):
    AB = math.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)
    AC = math.sqrt((C[0] - A[0])**2 + (C[1] - A[1])**2)
    BC = math.sqrt((C[0] - B[0])**2 + (C[1] - B[1])**2)
    angle = math.degrees(math.acos((AB**2 + AC**2 - BC**2) / (2 * AB * AC)))
    return int(angle)

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


def main():

    model = YOLO('models/yolov8n-pose.pt')  # load an official model
    cv2.namedWindow("ROI",cv2.WINDOW_NORMAL)

    source_video_path = "videos/demo.mp4"
    video_saving_path = source_video_path[:len(source_video_path)-4:]+"with_keypoinst.mp4"

    video_cap=cv2.VideoCapture(source_video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    desired_fps = 35

    if video_write:
        video = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,fps, (width,height))

    count=0
    while video_cap.isOpened():

        ret,frame=video_cap.read()

        if not ret:
            break
        #ADJUST FPS
        count +=1
        if count % 10 != 0:
            continue

        results = model.predict(source=frame,device="cpu")
        print(len(results))

        frame = results[0].plot()

        #plotted = results[0].plot()

        if not video_write:
            cv2.imshow("ROI",frame)
        if video_write:
            print(f"frame {count} writing")
            video.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    
    video_cap.release()
    if video_write:
        video.release()
    cv2.destroyAllWindows()
    print("process done")

if __name__ == "__main__":
    main()