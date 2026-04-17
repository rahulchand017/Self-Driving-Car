import os
import sys
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.inference.run_steering_angle_prediction import SteeringAnglePredictor
from src.inference.run_segmentation_obj_det import ImageSegmentation


class SelfDrivingCarSimulator:
    def __init__(self, steering_predictor, image_segmentation, dataset_dir: str):
        self.steering_predictor = steering_predictor
        self.image_segmentation = image_segmentation
        self.dataset_dir = dataset_dir

        # load steering wheel image for visualization
        wheel_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'steering_wheel_image.jpg'
        )
        self.steering_wheel_img = cv2.imread(wheel_path, 0)
        self.rows, self.cols = self.steering_wheel_img.shape

        # smoothing - prevents the wheel from jerking around
        self.smoothed_angle = 0.0

        # load dataset frame list in order (option C - play dataset as video)
        data_txt = os.path.join(dataset_dir, 'data.txt')
        self.frame_paths = []
        with open(data_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    self.frame_paths.append(
                        os.path.join(dataset_dir, parts[0])
                    )

        print(f"loaded {len(self.frame_paths)} frames for simulation")

    def _rotate_steering_wheel(self, angle_deg: float) -> np.ndarray:
        # smooth the angle so the wheel doesn't jerk
        # 0.2 weight on new angle, 0.8 on previous - adjust for more/less smoothing
        self.smoothed_angle += 0.2 * pow(abs(angle_deg - self.smoothed_angle), 2.0 / 3.0) * (
            1 if angle_deg > self.smoothed_angle else -1
        )

        # rotate the steering wheel image
        M = cv2.getRotationMatrix2D(
            (self.cols / 2, self.rows / 2), -self.smoothed_angle, 1
        )
        dst = cv2.warpAffine(self.steering_wheel_img, M, (self.cols, self.rows))
        return dst

    def start_simulation(self, frame_interval: float = 1/30):
        print("starting simulation, press Q to quit")
        print("three windows: Original Frame | Segmented Frame | Steering Wheel")

        for frame_path in self.frame_paths:
            # read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # predict steering angle
            angle_deg = self.steering_predictor.predict_angle(frame)

            # run segmentation and object detection
            segmented_frame = self.image_segmentation.process_frame(frame)

            # rotate steering wheel
            wheel = self._rotate_steering_wheel(angle_deg)

            # display info on original frame
            cv2.putText(
                frame,
                f"Predicted steering angle: {angle_deg:.2f} degrees",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # show the three windows
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Segmented Frame", segmented_frame)
            cv2.imshow("Steering Wheel", wheel)

            # press Q to quit
            if cv2.waitKey(int(frame_interval * 1000)) & 0xFF == ord('q'):
                print("simulation stopped by user")
                break

        cv2.destroyAllWindows()
        print("simulation ended")


if __name__ == "__main__":
    # paths - adjust if your folder structure is different
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

    STEERING_MODEL_PATH = os.path.join(
        BASE_DIR, 'saved_models', 'regression_model', 'model.ckpt'
    )
    LANE_MODEL_PATH = os.path.join(
        BASE_DIR, 'saved_models', 'lane_segmentation_model', 'best_yolo11_lane_segmentation.pt'
    )
    OBJECT_MODEL_PATH = os.path.join(
        BASE_DIR, 'saved_models', 'object_detection_model', 'yolo11s-seg.pt'
    )
    DATASET_DIR = os.path.join(BASE_DIR, 'data', 'driving_dataset')

    print("loading models...")
    steering_predictor = SteeringAnglePredictor(STEERING_MODEL_PATH)
    image_segmentation = ImageSegmentation(LANE_MODEL_PATH, OBJECT_MODEL_PATH)

    simulator = SelfDrivingCarSimulator(
        steering_predictor,
        image_segmentation,
        DATASET_DIR
    )

    try:
        simulator.start_simulation(frame_interval=1/30)
    finally:
        steering_predictor.close()
        print("all done")
