#TODO make polygons filled with transparant color

import torch
import cv2
import numpy as np
import time 
start_time = time.time()


# Load a model

# Predict with the model


source_video_path = "videos/demo.mp4"
video_saving_path = source_video_path[:len(source_video_path)-4:]+"_output.mp4"



video_write = False

cv2.namedWindow("xalil", cv2.WINDOW_NORMAL)

video_cap=cv2.VideoCapture(source_video_path)
fps = video_cap.get(cv2.CAP_PROP_FPS)
width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
desired_fps = 35

if video_write:
    result = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,fps, (width,height))


count=0
while video_cap.isOpened():

    ret,frame=video_cap.read()

    if not ret:
        break
    #ADJUST FPS
    count +=1
    if count % 100 != 0:
        continue


    cv2.imshow("xalil",frame)
    cv2.resizeWindow("xalil", 720, 540)
    if video_write:
        print(f"frame {count} writing")
        result.write(frame)
    if cv2.waitKey(30) == ord('q'):
        break


video_cap.release()
if video_write:
    result.release()
cv2.destroyAllWindows()
print("Execution time:", time.time() - start_time, "seconds")
print("process done")