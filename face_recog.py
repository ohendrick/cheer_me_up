# cheer me up app
# edited from https://towardsdatascience.com/the-ultimate-guide-to-emotion-recognition-from-facial-expressions-using-python-64e58d4324ff
# and https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
#
# This python code uses live video feed to look at facial expressions and attempts to segment the current emotion
# if the emotion is sad it will show a cheerful message in the output

from fer import FER
import matplotlib.pyplot as plt
#%matplotlib inline

import cv2

# define a video capture object
vid = cv2.VideoCapture(1)

while (True):



    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # EMO
    emo_detector = FER(mtcnn=True)
    # Capture all the emotions on the image
    captured_emotions = emo_detector.detect_emotions(frame)
    # Print all captured emotions with the image
    print(captured_emotions)
    plt.imshow(frame)

    # Use the top Emotion() function to call for the dominant emotion in the image
    dominant_emotion, emotion_score = emo_detector.top_emotion(frame)
    print(dominant_emotion)

    #show a message
    if dominant_emotion == "sad":
        print("YOU ARE A BEAUTIFUL PERSON, THINGS WILL BE OK")

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
