import os
import sys
import cv2
import numpy as np

# add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

from src.inference.run_steering_angle_prediction import SteeringAnglePredictor
from src.inference.run_segmentation_obj_det import ImageSegmentation


class SelfDrivingCarSimulator:
    def __init__(self, steering_predictor, image_segmentation,
                 dataset_dir: str):
        self.steering_predictor = steering_predictor
        self.image_segmentation = image_segmentation
        self.dataset_dir = dataset_dir

        # load steering wheel image
        wheel_path = os.path.join(PROJECT_ROOT, 'data', 'steering_wheel_image.jpg')
        wheel = cv2.imread(wheel_path, 0)
        if wheel is None:
            raise FileNotFoundError(f'steering wheel image not found at {wheel_path}')
        self.wheel_img = wheel
        self.rows, self.cols = wheel.shape[:2]

        # smoothed angle for stable wheel rotation
        self.smoothed_angle = 0.0

        # load frame paths from data.txt in order
        data_txt = os.path.join(dataset_dir, 'data.txt')
        self.frame_paths = []
        with open(data_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    self.frame_paths.append(
                        os.path.join(dataset_dir, parts[0])
                    )

        print(f'loaded {len(self.frame_paths)} frames')

    def _rotate_wheel(self, angle_deg: float) -> np.ndarray:
        # smooth the angle to avoid jerky wheel movement
        diff = abs(angle_deg - self.smoothed_angle)
        self.smoothed_angle += 0.2 * pow(diff, 2.0 / 3.0) * (
            1 if angle_deg > self.smoothed_angle else -1
        )

        M = cv2.getRotationMatrix2D(
            (self.cols / 2, self.rows / 2),
            -self.smoothed_angle, 1
        )
        return cv2.warpAffine(self.wheel_img, M, (self.cols, self.rows))

    def start_simulation(self, fps: int = 30):
        print('starting simulation...')
        print('press Q to quit')

        frame_delay = int(1000 / fps)

        for frame_path in self.frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            # predict steering angle
            angle_deg = self.steering_predictor.predict_angle(frame)

            # run segmentation
            segmented = self.image_segmentation.process_frame(frame)

            # rotate steering wheel
            wheel = self._rotate_wheel(angle_deg)

            # overlay angle text on original frame
            cv2.putText(
                frame,
                f'Steering: {angle_deg:.2f} deg',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

            # show three windows
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Segmented Frame', segmented)
            cv2.imshow('Steering Wheel', wheel)

            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                print('stopped by user')
                break

        cv2.destroyAllWindows()
        print('simulation ended')


if __name__ == '__main__':
    BASE = PROJECT_ROOT

    STEERING_MODEL = os.path.join(
        BASE, 'saved_models', 'regression_model', 'model.ckpt'
    )
    LANE_MODEL = os.path.join(
        BASE, 'saved_models', 'lane_segmentation_model',
        'best_yolo11_lane_segmentation.pt'
    )
    OBJECT_MODEL = os.path.join(
        BASE, 'saved_models', 'object_detection_model', 'yolo11s-seg.pt'
    )
    DATASET_DIR = os.path.join(BASE, 'data', 'driving_dataset')

    print('loading models...')
    steering = SteeringAnglePredictor(STEERING_MODEL)
    segmentation = ImageSegmentation(LANE_MODEL, OBJECT_MODEL)

    sim = SelfDrivingCarSimulator(steering, segmentation, DATASET_DIR)

    try:
        sim.start_simulation(fps=30)
    finally:
        steering.close()
        print('done')
