import numpy as np
import argparse
import cv2

# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
roiPts = []
inputMode = False

def selectROI(event, x, y, flags, parameters):
    '''
    grab ref to current frame, list of ROI, points whether in roi selection /
    '''
    global frame, roiPts, inputMode
    
    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)

def main():
    global frame, roiPts, inputMode
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    args = vars(ap.parse_args())
    
    # if vid path is not supplied, grab ref to camera
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
        
    # mouse callback
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)
    
    # initialize termination criteria for CamShift, indicating max of ten iterations per move
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    
    while True:
        # grab current frame
        (grabbed, frame) = camera.read()
        
        # check if we reached the end of the video
        if not grabbed:
            break
        
        # roi computation check
        if roiBox is not None:
            # convert the current frame to the HSV color space
            # and perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            
            # apply CamShift to the back projection, convert the
            # points to a bounding box, and then draw them
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.boxPoints(r))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
        # show frame if and record if a key is pressed
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # handle if "i" is pressed -> go into roi
        if key == ord("i") and len(roiPts) < 4:
            # indicate we are in input mode
            inputMode = True
            orig = frame.copy()

            # loop until 4 reference points are selected or any key is pressed to exit
            while len(roiPts) < 4:
                cv2.imshow("frame", frame)
                cv2.waitKey(0)
                
            # determine top left/right points
            roiPts = np.array(roiPts)
            s = roiPts.sum(axis=1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]
            
            # grab roi for bounding point and convert it to HSV
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # compute HSV histogram
            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (tl[0], tl[1], br[0], br[1])
        elif key == ord("q"):
            break
    
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
