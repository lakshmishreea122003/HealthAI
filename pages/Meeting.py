import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import mediapipe as mp
import pickle
import av

# load model
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
json_file = open("C:/Users/Lakshmi/Downloads/WebCam-Face-Emotion-Detection-Streamlit-main/WebCam-Face-Emotion-Detection-Streamlit-main/emotion_model1.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("C:/Users/Lakshmi/Downloads/WebCam-Face-Emotion-Detection-Streamlit-main/WebCam-Face-Emotion-Detection-Streamlit-main/emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier("C:/Users/Lakshmi/Downloads/WebCam-Face-Emotion-Detection-Streamlit-main/WebCam-Face-Emotion-Detection-Streamlit-main/haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")

emo="apple"
full = []

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img
    



# class VideoProcessor:
#     def recv(self, frame):
#         frame = frame.to_ndarray(format="bgr24")
#         #  Convert the image to grayscale
#         img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Initialize MediaPipe Hands
#         mp_hands = mp.solutions.hands
#         hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#         # Process the frame with MediaPipe Hands
#         results = hands.process(img_gray)

#         # Check for detected hands and draw landmarks
#         if results.multi_hand_landmarks:
#             for landmarks in results.multi_hand_landmarks:
#                 for landmark in landmarks.landmark:
#                     x, y, _ = frame.shape
#                     x_pos, y_pos = int(landmark.x * x), int(landmark.y * y)
#                     cv2.circle(frame, (x_pos, y_pos), 5, (0, 0, 255), -1)  # Draw a red circle at each hand landmark

#         # Release resources
#         hands.close()
#         # Convert the frame to RGB format (required by YOLOv5)
       
#         return av.VideoFrame.from_ndarray(frame, format='bgr24')   
# class Emoji_text(VideoProcessorBase): 
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         # Convert the image to grayscale
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Initialize MediaPipe Hands
#         mp_hands = mp.solutions.hands
#         hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#         # Process the frame with MediaPipe Hands
#         results = hands.process(img_gray)

#         # Check for detected hands and draw landmarks
#         if results.multi_hand_landmarks:
#             for landmarks in results.multi_hand_landmarks:
#                 for landmark in landmarks.landmark:
#                     x, y, _ = img.shape
#                     x_pos, y_pos = int(landmark.x * x), int(landmark.y * y)
#                     cv2.circle(img, (x_pos, y_pos), 5, (0, 0, 255), -1)  # Draw a red circle at each hand landmark

#         # Release resources
#         hands.close()

#         return av.VideoFrame(img) 
    # def transform(self,frame):
    #     model_dict = pickle.load(open(r"C:\Users\Lakshmi\PycharmProjects\mediapipe-practice\SilentBridge\hand_gesture2\model.p", 'rb'))
    #     model = model_dict['model']

    #     mp_hands = mp.solutions.hands
    #     mp_drawing = mp.solutions.drawing_utils
    #     mp_drawing_styles = mp.solutions.drawing_styles

    #     hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    #     labels_dict = {0: '0', 1: '1', 2: '2',3:'3'}

    #     data_aux = []
    #     x_ = []
    #     y_ = []

    #     H, W, _ = frame.shape

    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #     results = hands.process(frame_rgb)
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             mp_drawing.draw_landmarks(
    #             frame,  # image to draw
    #             hand_landmarks,  # model output
    #             mp_hands.HAND_CONNECTIONS,  # hand connections
    #             mp_drawing_styles.get_default_hand_landmarks_style(),
    #             mp_drawing_styles.get_default_hand_connections_style())

    #         for hand_landmarks in results.multi_hand_landmarks:
    #             for i in range(len(hand_landmarks.landmark)):
    #                x = hand_landmarks.landmark[i].x
    #                y = hand_landmarks.landmark[i].y

    #                x_.append(x)
    #                y_.append(y)

    #             for i in range(len(hand_landmarks.landmark)):
    #                x = hand_landmarks.landmark[i].x
    #                y = hand_landmarks.landmark[i].y
    #                data_aux.append(x - min(x_))
    #                data_aux.append(y - min(y_))

    #         x1 = int(min(x_) * W) - 10
    #         y1 = int(min(y_) * H) - 10

    #         x2 = int(max(x_) * W) - 10
    #         y2 = int(max(y_) * H) - 10

    #         prediction = model.predict([np.asarray(data_aux)])

    #         predicted_character = labels_dict[int(prediction[0])]

    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    #         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
    #                 cv2.LINE_AA)
            
    #         img = frame.to_ndarray(format="bgr24")
           

    #         return img
   




def main():
    st.header("emoji+sign_to _text")
    webrtc_streamer(key="emoji", video_transformer_factory=VideoTransformer)
    # # Face Analysis Application #
    # st.title("Real Time Face Emotion Detection Application")
    # activiteis = ["emoji+sign_to_text", "sign_to_voice","text_to_voice","voice_to_text/sign",'emotion']
    # choice = st.sidebar.selectbox("Select Task", activiteis)
    
    # if choice == "emoji+sign_to _text":
    #     st.header("emoji+sign_to _text")
    #     webrtc_streamer(key="emoji", video_transformer_factory=Emoji_text)
        
    # elif choice == "emotion":
    #     st.header("Webcam Live Feed")
    #     st.write("Click on start to use webcam and detect your face emotion")
    #     webrtc_streamer(key="example", video_transformer_factory=Emoji_text)


    # elif choice == "About":
    #     st.subheader("About this app")
    #     html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
    #                                 <h4 style="color:white;text-align:center;">
    #                                 Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
    #                                 </div>
    #                                 </br>"""
    #     st.markdown(html_temp_about1, unsafe_allow_html=True)

    #     html_temp4 = """
    #                          		<div style="background-color:#98AFC7;padding:10px">
    #                          		<h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at Mohammad.juned.z.khan@gmail.com. </h4>
    #                          		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
    #                          		</div>
    #                          		<br></br>
    #                          		<br></br>"""

    #     st.markdown(html_temp4, unsafe_allow_html=True)

    # else:
    #     pass


if __name__ == "__main__":
    main()



