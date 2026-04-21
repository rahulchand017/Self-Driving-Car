import cv2
import numpy as np
import tensorflow as tf


class SteeringAnglePredictor:
    def __init__(self, model_path: str):
        # load the trained checkpoint using tf2 compat v1
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session()
            saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
            saver.restore(self.sess, model_path)

            # get input/output tensors by name
            self.x = self.graph.get_tensor_by_name('Placeholder:0')
            self.keep_prob = self.graph.get_tensor_by_name('Placeholder_2:0')
            self.y = self.graph.get_tensor_by_name('Mul:0')

        print(f'steering model loaded from {model_path}')

    def predict_angle(self, image: np.ndarray) -> float:
        # preprocess exactly as done during training
        img_cropped = image[-150:]
        img_resized = cv2.resize(img_cropped, (200, 66))
        img_normalized = img_resized / 255.0
        img_batch = [img_normalized]

        with self.graph.as_default():
            angle_rad = self.sess.run(
                self.y,
                feed_dict={self.x: img_batch, self.keep_prob: 1.0}
            )[0][0]

        angle_deg = float(angle_rad * 180.0 / np.pi)
        return angle_deg

    def close(self):
        self.sess.close()
