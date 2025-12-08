#!/usr/bin/env python3
"""
Standalone video runner for the Localizer2D pipeline.
Reads frames from a video (or webcam), applies YOLO + optical-flow tracking,
and writes the annotated frames back out via OpenCV.
"""

import time
import os
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO


import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

from cv_bridge import CvBridge 



class VideoLocalizer:
    """ROS-free replica of Localizer2D that runs purely on OpenCV video I/O."""

    # Optical flow 파라미터
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,  # maxLevel>0 enables pyramid LK optical flow
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    

    def __init__(self, model_path="yolo11s.pt", yolo_interval=5):
        # COCO class id -> label
        self.class_mapping = {
            32: "ball",
            41: "drone",  # original "cup" label repurposed for drone
        }

        ultralytics.checks()
        self.model = YOLO(model_path)
        # Warmup pass for stable first inference.
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

        self.prev_gray = None
        # track: {"id", "bbox", "points", "class_id", "miss_count"}
        self.tracks = []
        self.frame_count = 0  # 프레임 카운터
        self.yolo_interval = yolo_interval  # 프레임 간격
        self.yolo_interval = yolo_interval
        self.next_track_id = 0
        self.match_iou = 0.25
        self.redundant_iou = 0.2

        # YOLO에서 여러 번 연속으로 못 봐도 optical flow로 유지할지 결정할 때 사용
        self.max_missed = 5

    # -----------------------------
    #   Low-level helper methods
    # -----------------------------
    def _extract_points(self, gray, bbox):
        """Extract optical-flow keypoints inside bbox."""
        x1, y1, x2, y2 = bbox
        mask = np.zeros_like(gray)
        mask[int(y1): int(y2), int(x1): int(x2)] = 255
        pts = cv2.goodFeaturesToTrack(
            gray, mask=mask, maxCorners=50, qualityLevel=0.01, minDistance=3
        )
        return pts

    def _optical_flow_predict(self, prev_gray, curr_gray, tracks):
        """Shift existing tracks using Lucas-Kanade optical flow prediction.
        기존 track dict에 들어 있던 필드(id, class_id, miss_count 등)를
        유지한 채 bbox/points만 업데이트합니다.
        """
        updated_tracks = []

        for t in tracks:
            pts = t.get("points", None)
            if pts is None or len(pts) < 3:
                continue

            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts, None, **self.lk_params
            )

            if new_pts is None or status is None:
                continue

            good_old = pts[status == 1]
            good_new = new_pts[status == 1]

            if len(good_new) < 3:
                continue

            shift = np.median(good_new - good_old, axis=0)
            dx, dy = shift

            x1, y1, x2, y2 = t["bbox"]
            new_bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

            new_track = t.copy()
            new_track["bbox"] = new_bbox
            new_track["points"] = good_new.reshape(-1, 1, 2)

            updated_tracks.append(new_track)

        return updated_tracks

    def _compute_iou(self, box_a, box_b):
        """IoU helper for suppression/matching."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h

        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _blend_boxes(self, box_a, box_b, alpha=0.6):
        """Smooth bounding boxes with EMA-like blending."""
        return [
            (1 - alpha) * box_a[0] + alpha * box_b[0],
            (1 - alpha) * box_a[1] + alpha * box_b[1],
            (1 - alpha) * box_a[2] + alpha * box_b[2],
            (1 - alpha) * box_a[3] + alpha * box_b[3],
        ]

    # -----------------------------
    #   YOLO & track helpers
    # -----------------------------
    def _run_yolo(self, frame):
        """Run YOLO and return detections."""
        results = self.model(
            frame,
            classes=list(self.class_mapping.keys()),
            conf=0.1,
            verbose=False,
        )[0]

        detections = []
        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_id": cls_id,
                        "score": score,
                    }
                )

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    def _is_near_edge(self, bbox, img_shape, margin_ratio=0.05):
        """bbox가 화면 가장자리 근처에 있는지 여부."""
        h, w = img_shape[:2]
        x1, y1, x2, y2 = bbox

        margin_x = w * margin_ratio
        margin_y = h * margin_ratio

        # 완전히 화면 밖이면 edge로 간주
        if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
            return True

        # 화면 안이지만, 가장자리와 너무 가까우면 edge 근처
        if x1 < margin_x or y1 < margin_y or x2 > w - margin_x or y2 > h - margin_y:
            return True

        return False

    def _format_output(self):
        """현재 self.tracks를 (detections, counts) 형식으로 변환."""
        output_detections = []
        counts = {label: 0 for label in self.class_mapping.values()}

        for t in self.tracks:
            x1, y1, x2, y2 = t["bbox"]
            label = self.class_mapping.get(t["class_id"], "unknown")
            counts[label] = counts.get(label, 0) + 1
            output_detections.append(
                {
                    "id": t["id"],
                    "label": label,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        return output_detections, counts

    # -----------------------------
    #   Core per-frame processing
    # -----------------------------
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        # 1) optical flow로 기존 트랙 예측
        if self.prev_gray is not None and len(self.tracks) > 0:
            predicted_tracks = self._optical_flow_predict(
                self.prev_gray, gray, self.tracks
            )
        else:
            predicted_tracks = []

        # 2) 프레임 간격 기반으로 YOLO 실행 여부 결정
        run_yolo = (
            (self.frame_count % self.yolo_interval == 0)
            or (len(predicted_tracks) == 0)
        )

        if not run_yolo:
            self.tracks = predicted_tracks
            self.prev_gray = gray.copy()
            return self._format_output()

        # 3) YOLO 실행
        detections = self._run_yolo(frame)

        updated_tracks = []
        assigned_det_indices = set()
        assigned_track_ids = set()

        # 4) IoU 매트릭스 기반 매칭
        matches = []
        for det_idx, det in enumerate(detections):
            for track in predicted_tracks:
                if track["class_id"] != det["class_id"]:
                    continue
                iou_val = self._compute_iou(track["bbox"], det["bbox"])
                if iou_val >= self.match_iou:
                    matches.append((iou_val, det_idx, track))

        matches.sort(key=lambda x: x[0], reverse=True)

        for iou_val, det_idx, track in matches:
            if det_idx in assigned_det_indices:
                continue
            if track["id"] in assigned_track_ids:
                continue

            det = detections[det_idx]
            blended = self._blend_boxes(track["bbox"], det["bbox"])
            pts = self._extract_points(gray, blended)

            track["bbox"] = blended
            track["points"] = pts
            track["miss_count"] = 0

            assigned_det_indices.add(det_idx)
            assigned_track_ids.add(track["id"])
            updated_tracks.append(track)

        # 5) 새 트랙 생성
        for det_idx, det in enumerate(detections):
            if det_idx in assigned_det_indices:
                continue

            det_bbox = det["bbox"]
            cls_id = det["class_id"]

            redundant = False
            for track in updated_tracks:
                if track["class_id"] != cls_id:
                    continue
                if self._compute_iou(track["bbox"], det_bbox) > self.redundant_iou:
                    redundant = True
                    break

            if not redundant:
                for track in predicted_tracks:
                    if track["id"] in assigned_track_ids:
                        continue
                    if track["class_id"] != cls_id:
                        continue
                    if self._compute_iou(track["bbox"], det_bbox) > self.redundant_iou:
                        redundant = True
                        break

            if redundant:
                continue

            new_track = {
                "id": self.next_track_id,
                "bbox": det_bbox,
                "points": self._extract_points(gray, det_bbox),
                "class_id": cls_id,
                "miss_count": 0,
            }
            self.next_track_id += 1
            updated_tracks.append(new_track)

        # 6) unmatched tracks 처리
        img_shape = gray.shape
        for track in predicted_tracks:
            if track["id"] in assigned_track_ids:
                continue

            dominated = False
            for ut in updated_tracks:
                if ut["class_id"] != track["class_id"]:
                    continue
                if self._compute_iou(ut["bbox"], track["bbox"]) > self.redundant_iou:
                    dominated = True
                    break
            if dominated:
                continue

            if not self._is_near_edge(track["bbox"], img_shape):
                track["miss_count"] = track.get("miss_count", 0) + 1
                if track["miss_count"] <= self.max_missed:
                    updated_tracks.append(track)

        self.tracks = updated_tracks
        self.prev_gray = gray.copy()
        return self._format_output()

    # -----------------------------
    #   Video loop & drawing
    # -----------------------------
    def process_video(self, source, output_path=None, display=True):
        """Run the detector over a cv2.VideoCapture source."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        print(f"[INFO] Input video: {fps:.2f} FPS, {width}x{height}")

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Press ESC to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections, counts = self.process_frame(frame)
            annotated = self.draw_detections(frame.copy(), detections, counts)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Localizer2D Video", annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    def draw_detections(self, frame, detections, counts):
        """Overlay bounding boxes and counts on the frame."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            color = (0, 255, 0) if det["label"] == "ball" else (255, 0, 0)
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(
                frame,
                det["label"],
                (p1[0], p1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        y = 25
        for label, count in counts.items():
            cv2.putText(
                frame,
                f"{label}: {count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            y += 25

        return frame




class Localizer2D(Node):
    def __init__(self):
        super().__init__('localizer_2d')
        self.get_logger().info("localizer 2d starting")

        self.image_save = False
        self.save_interval = 1
        self.max_images = 1000
        self.total_frame_count = 0
        self.saved_image_count = 0

        self.save_dir = 'captured_images'

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f"Created directory: {self.save_dir}")

        # 2. QoS setup
        qos_profile = qos_profile_sensor_data

        # 3. Subscribe, Publish
        self.subscription = self.create_subscription(
            Image,
            '/anafi/camera/image',
            self.image_callback,
            qos_profile)


        self.pub_detection = self.create_publisher(
            Detection2DArray, 'detections_bb', qos_profile_sensor_data)

        self.br = CvBridge()

        self.localizer = VideoLocalizer(model_path="yolo11s.pt", yolo_interval=5)
        
        self.get_logger().info("VideoLocalizer ready")


    def image_callback(self, msg):
        try:
            current_frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.total_frame_count += 1
        if ((self.image_save) and (self.saved_image_count < self.max_images) and
            (self.total_frame_count % self.save_interval == 0)):
            file_name = f"image_{self.saved_image_count + 1:03d}.png"
            save_path = os.path.join(self.save_dir, file_name)
            cv2.imwrite(save_path, current_frame)
            self.saved_image_count += 1

        detections, counts = self.localizer.process_frame(current_frame)

        # Publish 결과
        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2

            detection = Detection2D()
            detection.header = msg.header

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det["label"]
            hypothesis.hypothesis.score = 1.0
            detection.results.append(hypothesis)

            detection.bbox.center.position.x = float(x_center)
            detection.bbox.center.position.y = float(y_center)
            detection.bbox.size_x = float(w)
            detection.bbox.size_y = float(h)

            # track ID를 id 필드에 저장 (Detection2D에 id 필드가 있다면)
            detection.id = str(det.get("id", -1))

            detections_msg.detections.append(detection)

        self.get_logger().info(f"Published ball: {counts.get('ball', 0)}, drone: {counts.get('drone', 0)}")
        self.pub_detection.publish(detections_msg)




def main(args=None):
    rclpy.init(args=args)

    node = Localizer2D()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
