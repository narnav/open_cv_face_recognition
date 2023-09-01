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

# Dataset preparation
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

trainset = np.concatenate((face_dataset, face_labels), axis=1)

font = cv2.FONT_HERSHEY_SIMPLEX

# KNN functions (distance and knn)
def distance(v1, v2):
    # Euclidean distance
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

class VideoApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.layout = BoxLayout(orientation='vertical')

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

        Clock.schedule_interval(self.update, 1.0 / 30.0)

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
        self.signup_with_opencv()

    def signup_with_opencv(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

        skip = 0
        face_data = []

        file_name = input("Enter your name: ")

        while True:
            ret, frame = cap.read()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if ret == False:
                continue

            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            if len(faces) == 0:
                continue

            k = 1

            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

            skip += 1

            for face in faces[:1]:
                x, y, w, h = face

                offset = 5
                face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
                face_selection = cv2.resize(face_offset, (100, 100))

                if skip % 10 == 0:
                    face_data.append(face_selection)
                    print(len(face_data))

                cv2.imshow(str(k), face_selection)
                k += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("faces", frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                break
            if len(face_data) > 30:
                break

        face_data = np.array(face_data)
        face_data = face_data.reshape((face_data.shape[0], -1))
        print(face_data.shape)

        np.save(dataset_path + file_name, face_data)
        print("Dataset saved at : {}".format(dataset_path + file_name + '.npy'))

        cap.release()
        cv2.destroyAllWindows()

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for face in faces:
                x, y, w, h = face

                offset = 5
                face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
                face_section = cv2.resize(face_section, (100, 100))

                out = knn(trainset, face_section.flatten())
                recognized_name = names[int(out)]

                cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                self.recognized_name = recognized_name

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
