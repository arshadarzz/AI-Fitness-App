import cv2
import time
import os
from posemodule import poseDetector  # Make sure to import your poseDetector class

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

    pTime = 0
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos', 'squats1.mp4')
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    detector = poseDetector()

    squat_count = 0
    squat_stage = None  # None, 'down', 'up'

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmlist = detector.getPosition(img, draw=False)

        if len(lmlist) != 0:
            left_hip = lmlist[23][2]  # y-coordinate of left hip
            right_hip = lmlist[24][2]  # y-coordinate of right hip
            average_hip = (left_hip + right_hip) / 2  # average hip position

            # Check squat stage
            if average_hip < 200:  # Adjust this value based on your video
                squat_stage = 'down'
            if average_hip > 250 and squat_stage == 'down':  # Adjust this value based on your video
                squat_stage = 'up'
                squat_count += 1
                print(f'Squat Count: {squat_count}')

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Resize and show the image
        img = cv2.resize(img, (1100, 1100))
        cv2.imshow("Image", img)

        # Wait for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Ensure all OpenCV windows are destroyed properly

if __name__ == "__main__":
    main()