import cv2
import numpy as np
from ultralytics import YOLO


class ImageSegmentation:
    def __init__(self, lane_model_path: str, object_model_path: str):
        # lane model is our custom trained yolo on the roboflow dataset
        self.lane_model = YOLO(lane_model_path)
        # object model is pretrained yolo for cars, pedestrians etc
        self.object_model = YOLO(object_model_path)
        self.conf_threshold = 0.3

        # colors for lane classes (lane1 and lane2)
        self.lane_colors = [
            (0, 255, 255),   # lane1 - yellow
            (0, 165, 255),   # lane2 - orange
        ]

        # colors for object detection classes (using colorsys for variety)
        import colorsys
        self.object_colors = []
        for i in range(80):
            hue = i / 80.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            self.object_colors.append((int(b * 255), int(g * 255), int(r * 255)))

        print(f"lane model loaded: {lane_model_path}")
        print(f"object model loaded: {object_model_path}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # create overlay for drawing on
        overlay = frame.copy()

        # run lane segmentation first
        lane_results = self.lane_model.predict(
            frame, conf=self.conf_threshold, verbose=False
        )[0]
        overlay = self._draw_results(
            overlay, lane_results,
            self.lane_colors,
            show_conf=False,
            is_lane=True
        )

        # then run object detection on top
        obj_results = self.object_model.predict(
            frame, conf=self.conf_threshold, verbose=False
        )[0]
        overlay = self._draw_results(
            overlay, obj_results,
            self.object_colors,
            show_conf=True,
            is_lane=False
        )

        return overlay

    def _draw_results(self, overlay, results, colors, show_conf, is_lane):
        if results.masks is None:
            return overlay

        class_names = results.names

        for mask, box in zip(results.masks.xy, results.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_names[class_id]

            if is_lane:
                color = colors[class_id % len(colors)]
            else:
                color = colors[class_id % len(colors)]

            # draw segmentation mask
            if len(mask) > 0:
                mask_pts = mask.astype(np.int32)
                cv2.fillPoly(overlay, [mask_pts], color)

            # draw bounding box and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            if show_conf:
                label = f"{class_name} {confidence:.2f}"
            else:
                label = class_name
            cv2.putText(overlay, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # blend overlay with original for semi-transparent effect
        result = cv2.addWeighted(overlay, 0.5, overlay, 0.5, 0)
        return result
