import cv2
import mediapipe as mp
import pyautogui
import tensorflow as tf

def test_setup():
    print("Checking dependencies...")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"MediaPipe version: {mp.__version__}")
    print(f"PyAutoGUI version: {pyautogui.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print("All good!")

if __name__ == "__main__":
    test_setup()
