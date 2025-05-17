import cv2
import mediapipe as mp
import os
import time
import posemodule as pm

def pushup(n):
    pTime = 0
    # Path to the reference push-up video
    video_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos', 'pushup2.mp4')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0, 0

    detector = pm.poseDetector()
    count = 0
    f = 0
    time.sleep(5)  # Initial delay to prepare

    # Open the camera for live detection
    camera_cap = cv2.VideoCapture(0)
    if not camera_cap.isOpened():
        print("Error: Could not open the camera")
        return 0, 0

    while camera_cap.isOpened() and count < n:
        # Read frame from the camera
        success, img = camera_cap.read()
        if not success or img is None:
            print("Error: Frame not read successfully, exiting...")
            break

        # Read frame from the reference video
        reference_success, ref_img = cap.read()
        if not reference_success:
            print("Warning: Could not read a frame from the reference video.")
            ref_img = cv2.resize(ref_img, (800, 800)) if ref_img is not None else None

        img = detector.findPose(img)
        lmlist = detector.getPosition(img, draw=False)

        if len(lmlist) != 0:
            # Marking specific landmarks (elbow and shoulder)
            cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 10, (0, 0, 255), cv2.FILLED)  # Elbow
            cv2.circle(img, (lmlist[0][1], lmlist[0][2]), 10, (0, 0, 255), cv2.FILLED)  # Shoulder
            y1 = lmlist[14][2]  # Elbow Y coordinate
            y2 = lmlist[0][2]   # Shoulder Y coordinate
            
            length = y2 - y1
            
            if length >= 0 and f == 0:
                f = 1
            elif length < -50 and f == 1:
                f = 0
                count += 1

            cTime = time.time()
            fps = 1 / (cTime - pTime) if cTime != pTime else 0
            pTime = cTime

            # Display total push-ups and calories burned
            cv2.putText(img, f"Total Number of Push-ups: {count}", (70, 250), cv2.FONT_HERSHEY_DUPLEX, 3, (60, 100, 255), 3)
            cv2.putText(img, f"Calories Burned: {int(count * 0.29)}", (70, 350), cv2.FONT_HERSHEY_DUPLEX, 3, (60, 100, 255), 3)

            # Resize and display the camera image
            img = cv2.resize(img, (800, 800))
            cv2.imshow("Push-up Tracker", img)

            # Display the reference video frame for comparison
            if ref_img is not None:
                ref_img = cv2.resize(ref_img, (800, 800))
                cv2.imshow("Reference Push-up", ref_img)

            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= n:
                break

    camera_cap.release()
    cap.release()
    cv2.destroyAllWindows()
    calories = 0.29 * count  # Assuming each push-up burns approximately 0.29 calories

    return count, calories
