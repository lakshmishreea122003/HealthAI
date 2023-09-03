import cv2

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera (you can change this if you have multiple cameras)

# Define the coordinates of the rectangle (x, y, width, height)
rectangle = (100, 100, 200, 200)  # Modify these values as needed
count =0
while True:
    ret, frame = cap.read()  # Read a frame from the video feed

    if not ret:
        break

    x, y, w, h = rectangle
    hand_roi = frame[y:y + h, x:x + w]  # Crop the frame to the specified rectangle

    # Display the cropped frame with the rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video Feed", frame)
    cv2.imshow("Cropped Hand", hand_roi)
    
    # Save the cropped frame as an image (you can use a different file format if needed)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f'D:/llm projects/SilentBridge-streamlit/hand_gesture2/gestures/3/{count}.jpg', hand_roi)
        print("Hand image saved as 'hand_image.jpg'")
        count+=1
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


