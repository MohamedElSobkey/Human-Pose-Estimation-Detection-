#import packages
import cv2
import mediapipe as mp 
import numpy as np 

# initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()



# take video input for pose detection
# you can put here video of your choice
#cap = cv2.VideoCapture('H2.mp4')
cap = cv2.VideoCapture(0)

# take live camera  input for pose detection
# cap = cv2.VideoCapture(0)

# read each frame/image from capture object
while True :
    ret , video = cap.read()
    # resize image/frame so we can accommodate it on our screen
    video = cv2.resize(video , (600, 400))
    
    # do Pose detection
    results = pose.process(video)
    
    # draw the detected pose on original video/ live stream
    mp_draw.draw_landmarks(video, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
                           
    
    # Display pose on original video/live stream
    cv2.imshow("Pose Estimation", video)
    
    
    # Extract and draw pose on plain white video
    h, w, c = video.shape   # get shape of original frame
    opvideo = np.zeros([h, w, c]) # create blank video with original frame size
    opvideo.fill(0)# set white background. put 0 if you want to make it black
    
    
    
    # draw extracted pose on black white image
    mp_draw.draw_landmarks(opvideo, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    
    # display extracted pose on blank images
    cv2.imshow("Extracted Pose", opvideo)
    
    
    
    # print all landmarks
    print(results.pose_landmarks)
    
    
    cv2.waitKey(1)
    
    