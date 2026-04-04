import cv2 as cv


class Camera:
    def __init__(self, device_index=1):
        self.cap = cv.VideoCapture(device_index)
        if not self.cap.isOpened():
            # Fallback to default camera
            self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('frame', gray)
            if cv.waitKey(1) == ord('q'):
                break
        self.cap.release()
        cv.destroyAllWindows()
