import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from src.models import model


class SteeringAnglePredictor:
    def __init__(self, model_path: str):
        # load the trained tensorflow v1 checkpoint
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_path)
        print(f"steering model loaded from {model_path}")

    def predict_angle(self, image: np.ndarray) -> float:
        # image should be a raw bgr frame from opencv
        # we do the same preprocessing as training here
        img_cropped = image[-150:]
        img_resized = cv2.resize(img_cropped, (200, 66))
        img_normalized = img_resized / 255.0
        img_batch = [img_normalized]

        with self.sess.as_default():
            angle_rad = model.y.eval(
                feed_dict={model.x: img_batch, model.keep_prob: 1.0}
            )[0][0]

        # convert radians back to degrees for display
        angle_deg = float(angle_rad * 180.0 / np.pi)
        return angle_deg

    def close(self):
        self.sess.close()
