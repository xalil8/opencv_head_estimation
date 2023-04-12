#TODO make polygons filled with transparant color

import torch
import cv2
import numpy as np
import time 
from ultralytics import YOLO


# Load a model
model = YOLO('yolov8s-pose.pt')  # load an official model

# Predict with the model

cv2.namedWindow("ROI",cv2.WINDOW_NORMAL)
source_video_path = "demo.mp4"
video_saving_path = source_video_path[:len(source_video_path)-4:]+"_output.mp4"

video_write = False
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
    if count % 1 != 0:
        continue

    results = model.predict(source=frame,device="mps")


    for result in results:
        result = result.cpu().numpy()
        keypoints = result.keypoints
        for keypoint in keypoints:
            3print(keypoint)
            x1,y1,lenght = keypoint
            print


    cv2.imshow("ROI",frame)
    if video_write:
        print(f"frame {count} writing")
        video.write(frame)
    if cv2.waitKey(30) == ord('q'):
        break


video_cap.release()
if video_write:
    video.release()
cv2.destroyAllWindows()
print("process done")