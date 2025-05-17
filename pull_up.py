import cv2
import mediapipe as mp
import os
import time
import posemodule as pm

def pullup(n):
    pTime = 0
    # Define the path to the reference video
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos', 'pullup3.mp4')
    
    # Open the live camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera")
        return 0, 0

    # Open the reference video
    reference_video = cv2.VideoCapture(path)
    if not reference_video.isOpened():
        print(f"Error: Could not open reference video {path}")
        return 0, 0

    detector = pm.poseDetector()
    count = 0
    f = 0
    time.sleep(2)  # Initial delay to prepare

    # Prepare to read frames from the reference video
    reference_frames = []
    while True:
        success, ref_frame = reference_video.read()
        if not success:
            break
        reference_frames.append(ref_frame)

    # Main loop for processing the live feed
    while cap.isOpened() and count < n:
        success, img = cap.read()
        if not success or img is None:
            print("Error: Frame not read successfully, exiting...")
            break

        # Process the current frame from the camera
        img = detector.findPose(img)
        lmlist = detector.getPosition(img, draw=False)

        if len(lmlist) != 0:
            # Extract shoulder and wrist positions from the live feed
            wrist_live = lmlist[3][2]  # Wrist Y coordinate
            shoulder_live = lmlist[20][2]  # Shoulder Y coordinate
            length_live = shoulder_live - wrist_live

            # Analyze reference frames to get a matching position
            for ref_frame in reference_frames:
                ref_img = detector.findPose(ref_frame)
                ref_lmlist = detector.getPosition(ref_img, draw=False)

                if len(ref_lmlist) != 0:
                    # Extract shoulder and wrist positions from the reference
                    wrist_ref = ref_lmlist[3][2]  # Reference wrist Y coordinate
                    shoulder_ref = ref_lmlist[20][2]  # Reference shoulder Y coordinate
                    length_ref = shoulder_ref - wrist_ref

                    # Check if the current position is similar to the reference
                    if abs(length_live - length_ref) < 50:  # Threshold for position matching
                        # Counting logic
                        if length_live >= 0 and f == 0:
                            f = 1
                        elif length_live < 0 and f == 1:
                            f = 0
                            count += 1
                            break  # Exit the loop once a match is found for counting

            cTime = time.time()
            fps = 1 / (cTime - pTime) if cTime != pTime else 0
            pTime = cTime

            # Display total pull-ups and calories burned
            cv2.putText(img, f"Total Number of Pull-ups: {count}", (50, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (60, 100, 255), 2)
            cv2.putText(img, f"Calories Burned: {count * 1}", (50, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (60, 100, 255), 2)

            # Resize and display the image
            img = cv2.resize(img, (600, 600))
            cv2.imshow("Pull-up Tracker", img)

            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= n:
                break

    cap.release()
    reference_video.release()
    cv2.destroyAllWindows()
    calories = 1 * count  # Assuming each pull-up burns 1 calorie

    return count, calories

# Example usage (adjust the number of pull-ups to detect)
# pullup(10)
