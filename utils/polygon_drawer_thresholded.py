import cv2
import numpy as np


video_path = "/Users/xalil/Desktop/opencv_head_estimation/videos/demo.mp4"


class PolygonDrawer:
    def __init__(self):
        self.polygon_coordinates = []
        cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('ROI', self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print("Current coordinates", x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click
            print(f"""[{x}, {y}] saved to polygon list""")
            self.polygon_coordinates.append([x, y])

    def main(self):

        video_cap = cv2.VideoCapture(video_path)

        count = 0
        paused = False
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            #ADJUST FPS
            count +=1
            if count % 10 != 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            threshold = cv2.cvtColor(threshold,cv2.COLOR_GRAY2BGR)
            if len(self.polygon_coordinates) > 1:
                points = np.array([self.polygon_coordinates])
                cv2.polylines(threshold, np.int32([points]), True, (0, 0, 255), 3)

            cv2.imshow("ROI", frame)

            if paused:
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(60)

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused



        video_cap.release()
        cv2.destroyAllWindows()

        print("Final coordinates\n", self.polygon_coordinates)


if __name__ == "__main__":
    pd = PolygonDrawer()
    pd.main()
