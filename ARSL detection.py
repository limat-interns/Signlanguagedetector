import cv2
from ultralytics import YOLO

# Load the model
model = YOLO(r"C:\Users\HARISH\OneDrive\Desktop\ArSL\ASL-main\ASL.pt")

# Open the camera
cap = cv2.VideoCapture(0)

# Setting width and height of the video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


while cap.isOpened():
    
    success, frame = cap.read()

    if success:
        
        results = model.track(frame, persist=True)

        
        annotated_frame = results[0].plot()
        
        cv2.imshow("ASL Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if 
        break

cap.release()
cv2.destroyAllWindows()



