import numpy as np
import cv2
import matplotlib as plt

def run_main():
    #To save the video later
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.avi ', fourcc, 20.0, (640, 480))

    cap = cv2.VideoCapture('video/thermal_stereo_mv.mp4')
    #cap = cv2.VideoCapture('video/WhatsApp.mp4')

    # Read the first frame of the video
    ret, frame = cap.read()
    height, width, channels = frame.shape
    print(height, width, channels)
    frame=frame[0:480,0:1280] #left
    #frame = frame[0:480,640:1280]  # right

    #feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)




    # Set the ROI (Region of Interest). Actually, this is a
    # rectangle of the building that we're tracking

    x, y = corners[22].ravel()
    c, r, w, h = x, y, 50, 50
    track_window = (c, r, w, h)

    # Create mask and normalized histogram
    roi = frame[r:r + h, c:c + w]
    #hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_roi=roi

    lower_grey = np.array([0, 255, 255])
    upper_grey = np.array([10, 255, 255])
    #mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))
    mask = cv2.inRange(hsv_roi, lower_grey , upper_grey)
    #roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [256], [0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)

    #grey = np.uint8([[[0, 0, 255]]])
    #hsv_grey = cv2.cvtColor(grey, cv2.COLOR_BGR2HSV)
    #print (hsv_grey)





    while True:
        ret, frame = cap.read()

        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        dst = cv2.calcBackProject([frame ], [0], roi_hist, [0, 256], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.putText(frame, 'Tracked', (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.CV_8UC1)#CV_AA)
        out.write(frame)
        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_main()
