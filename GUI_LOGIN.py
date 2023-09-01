import cv2
import numpy as np
import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

dataset_path = "./face_dataset/"
face_data = []
labels = []
class_id = 0
names = {}


# Dataset prepration
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path + fx)
		face_data.append(data_item)

		target = class_id * np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

# Example KNN function
def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from the test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get the top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find the label with the maximum frequency
    index = np.argmax(output[1])
    return output[0][index]

class VideoApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.layout = BoxLayout(orientation='vertical')

        # Double the height of the Image widget
        self.image = Image(size_hint=(1, 2))
        self.layout.add_widget(self.image)

        self.start_stop_button = Button(text="Start Video")
        self.start_stop_button.bind(on_press=self.toggle_video)
        self.layout.add_widget(self.start_stop_button)

        self.login_button = Button(text="Login")
        self.login_button.bind(on_press=self.login)
        self.layout.add_widget(self.login_button)

        self.signup_button = Button(text="Signup")
        self.signup_button.bind(on_press=self.signup)
        self.layout.add_widget(self.signup_button)

        self.label = Label(text="Recognized Face: ")
        self.layout.add_widget(self.label)

        self.recognized_name = ""

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update the video at 30 FPS

        return self.layout

    def toggle_video(self, instance):
        if self.capture.isOpened():
            self.capture.release()
            self.start_stop_button.text = "Start Video"
            self.recognized_name = ""
        else:
            self.capture = cv2.VideoCapture(0)
            self.start_stop_button.text = "Stop Video"

    def login(self, instance):
        # Implement your login logic here
        pass

    def signup(self, instance):
        # Implement your signup logic here
        pass

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for face in faces:
                x, y, w, h = face

                # Get the face ROI
                offset = 5
                face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
                face_section = cv2.resize(face_section, (100, 100))

                # Replace this with your KNN code for recognition
                out = knn(trainset, face_section.flatten())  # Replace with your actual training data
                recognized_name = names[int(out)]

                # Draw rectangle and recognized name on the original image
                cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                self.recognized_name = recognized_name

            # Convert the frame to RGB format for displaying in Kivy
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image.texture = self.texture_from_frame(frame_rgb)

        self.label.text = f"Recognized Face: {self.recognized_name}"

    def texture_from_frame(self, frame):
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
        return texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    VideoApp().run()
