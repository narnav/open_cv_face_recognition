from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture
import cv2
import numpy as np

class LoginSignupApp(App):
    def build(self):
        # Create the main layout
        main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        # Create a label and a TextInput for username
        username_label = Label(text="Username:")
        self.username_input = TextInput(multiline=False)

        # Create labels for login and signup
        login_label = Label(text="Login")
        signup_label = Label(text="Sign Up")

        # Create login and signup buttons
        login_button = Button(text="Login", size_hint=(None, None), size=(100, 50))
        signup_button = Button(text="Sign Up", size_hint=(None, None), size=(100, 50))

        # Bind button click events to functions
        login_button.bind(on_release=self.login)
        signup_button.bind(on_release=self.signup)

        # Create an Image widget for displaying captured images
        self.image_widget = KivyImage(allow_stretch=True)

        # Add widgets to the main layout
        main_layout.add_widget(username_label)
        main_layout.add_widget(self.username_input)
        main_layout.add_widget(login_label)
        main_layout.add_widget(login_button)
        main_layout.add_widget(signup_label)
        main_layout.add_widget(signup_button)
        main_layout.add_widget(self.image_widget)

        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        self.cap = cv2.VideoCapture(0)
        self.face_data = []
        self.dataset_path = "./face_dataset/"

        # Hide the OpenCV window
        cv2.namedWindow("faces", cv2.WND_PROP_VISIBLE)
        cv2.setWindowProperty("faces", cv2.WND_PROP_VISIBLE, cv2.WINDOW_FULLSCREEN)

        return main_layout

    def login(self, instance):
        # Handle login button click
        print("Login button clicked")

        while True:
            ret, frame = self.cap.read()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if ret == False:
                continue

            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            if len(faces) == 0:
                continue

            k = 1

            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

            for face in faces[:1]:
                x, y, w, h = face

                offset = 5
                face_offset = frame[y - offset : y + h + offset, x - offset : x + w + offset]
                face_selection = cv2.resize(face_offset, (100, 100))

                if len(self.face_data) < 30:
                    self.face_data.append(face_selection)
                    print(len(self.face_data))

                # Display the captured image in the Kivy Image widget
                texture = self._cvimage_to_texture(face_selection)
                self.image_widget.texture = texture

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show the video frame without creating a separate window
            cv2.imshow("faces", frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord("q"):
                break
            if len(self.face_data) >= 30:
                break

        self.face_data = np.array(self.face_data)
        self.face_data = self.face_data.reshape((self.face_data.shape[0], -1))
        print(self.face_data.shape)

        np.save(self.dataset_path + self.username_input.text, self.face_data)
        print("Dataset saved at : {}".format(self.dataset_path + self.username_input.text + ".npy"))

    def signup(self, instance):
        # Handle signup button click
        self.file_name = self.username_input.text
        self.login(instance)  # Call the login function to capture and save face data

    def _cvimage_to_texture(self, image):
        buf = image.tostring()
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture


if __name__ == "__main__":
    LoginSignupApp().run()
