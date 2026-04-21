import cv2
import numpy as np
import colorsys
from ultralytics import YOLO


class ImageSegmentation:
    def __init__(self, lane_model_path: str, object_model_path: str):
        self.lane_model = YOLO(lane_model_path)
        self.object_model = YOLO(object_model_path)
        self.conf_threshold = 0.3

        # lane colors - yellow for lane1, orange for lane2
        self.lane_colors = [
            (0, 255, 255),
            (0, 165, 255),
        ]

        # generate distinct colors for 80 coco object classes
        self.object_colors = []
        for i in range(80):
            h = i / 80.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
            self.object_colors.append(
                (int(b * 255), int(g * 255), int(r * 255))
            )

        print(f'lane model loaded: {lane_model_path}')
        print(f'object model loaded: {object_model_path}')

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()

        # run lane segmentation
        lane_results = self.lane_model.predict(
            frame, conf=self.conf_threshold, verbose=False
        )[0]
        overlay = self._draw_results(
            frame, overlay, lane_results, self.lane_colors,
            show_conf=False, is_lane=True
        )

        # run object detection on top
        obj_results = self.object_model.predict(
            frame, conf=self.conf_threshold, verbose=False
        )[0]
        overlay = self._draw_results(
            frame, overlay, obj_results, self.object_colors,
            show_conf=True, is_lane=False
        )

        return overlay

    def _draw_results(self,frame, overlay, results, colors,
                      show_conf, is_lane):
        if results.masks is None:
            return overlay

        class_names = results.names

        for mask, box in zip(results.masks.xy, results.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_names[class_id]
            color = colors[class_id % len(colors)]

            # draw filled segmentation mask
            if len(mask) > 0:
                mask_pts = mask.astype(np.int32)
                cv2.fillPoly(overlay, [mask_pts], color)

            # draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # draw label
            label = f'{class_name} {confidence:.2f}' if show_conf else class_name
            cv2.putText(overlay, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # blend for semi-transparent effect
        result = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
        return result
