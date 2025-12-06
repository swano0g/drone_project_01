#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

from cv_bridge import CvBridge 
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
import os
import sys
import time


# =======================================
# ==== Optical Flow 관련 함수 추가 ========
# =======================================

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

def extract_points(gray, bbox):
    """ bbox 영역 안에서 optical flow용 keypoint 추출 """
    x1, y1, x2, y2 = bbox
    mask = np.zeros_like(gray)
    mask[int(y1):int(y2), int(x1):int(x2)] = 255
    pts = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=50,
                                  qualityLevel=0.01, minDistance=3)
    return pts

def optical_flow_predict(prev_gray, curr_gray, tracks):
    """ 기존 track들의 bbox를 optical flow로 이동시키기 """
    updated_tracks = []

    for t in tracks:
        pts = t["points"]
        if pts is None or len(pts) < 3:
            continue

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **lk_params
        )

        good_old = pts[status == 1]
        good_new = new_pts[status == 1]

        if len(good_new) < 3:
            continue

        shift = np.median(good_new - good_old, axis=0)
        dx, dy = shift

        x1, y1, x2, y2 = t["bbox"]
        new_bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

        updated_tracks.append({
            "id": t["id"],
            "bbox": new_bbox,
            "points": good_new.reshape(-1,1,2),
            "class_id": t["class_id"]   # ==== class_id 유지 ====
        })

    return updated_tracks




# ==================================================
# 🔵 기존 Localizer2D 클래스 (최소 변경으로 통합)
# ==================================================

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

        
        # 4. YOLO 
        self.class_mapping = {
            73: "ball",
            41: "drone", # cup
        }

        self.br = CvBridge() 

        ultralytics.checks()
        self.model = YOLO("yolo11s.pt")
        self.model(np.zeros((640,640,3), dtype=np.uint8), verbose=False)

        self.get_logger().info("YOLO ready")

        # ==================================================
        # ==== Optical Flow tracking 저장 변수 추가 ========
        # ==================================================
        self.prev_gray = None
        self.tracks = []     # {"id", "bbox", "points", "class_id"}
        self.last_yolo_time = time.time()



    def image_callback(self, msg):
        try:
            current_frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        now = time.time()

        self.total_frame_count += 1
        if ((self.image_save) and (self.saved_image_count < self.max_images) and
            (self.total_frame_count % self.save_interval == 0)):
            file_name = f"image_{self.saved_image_count + 1:03d}.png"
            save_path = os.path.join(self.save_dir, file_name)
            cv2.imwrite(save_path, current_frame)
            self.saved_image_count += 1


        # ==================================================
        # ================= YOLO 갱신 조건 =================
        # ==================================================
        yolo_update = (now - self.last_yolo_time > 1.0)  # 1초마다 YOLO 실행

        if yolo_update:
            self.last_yolo_time = now

            results = self.model.track(
                current_frame,
                classes=list(self.class_mapping.keys()),
                conf=0.1,
                persist=True,
                verbose=False
            )

            new_tracks = []

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # YOLO bbox (xywh)
                    x_center, y_center, w, h = box.xywh[0].cpu().numpy()
                    x1 = x_center - w/2
                    y1 = y_center - h/2
                    x2 = x_center + w/2
                    y2 = y_center + h/2

                    cls_id = int(box.cls[0])
                    track_id = -1
                    if box.id is not None:
                        track_id = int(box.id[0])

                    # =======================================
                    # ==== 기존 track과 id 기반 매칭 ==========
                    # =======================================
                    existing = None
                    for t in self.tracks:
                        if t["id"] == track_id:
                            existing = t
                            break

                    # class_id 유지 or YOLO로 초기화
                    if existing is not None:
                        final_class_id = existing["class_id"]
                    else:
                        final_class_id = cls_id

                    pts = extract_points(gray, [x1, y1, x2, y2])

                    new_tracks.append({
                        "id": track_id,
                        "bbox": [x1, y1, x2, y2],
                        "points": pts,
                        "class_id": final_class_id
                    })

            # YOLO 프레임에서는 optical flow 대신 YOLO 결과로 덮어씀
            self.tracks = new_tracks

        else:
            # ==================================================
            # ========== Optical Flow로 bbox 이동 예측 =========
            # ==================================================
            if self.prev_gray is not None and len(self.tracks) > 0:
                self.tracks = optical_flow_predict(self.prev_gray, gray, self.tracks)



        # ==================================================
        # ================== Publish 결과 ==================
        # ==================================================

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        ball = 0
        drone = 0

        for t in self.tracks:
            x1, y1, x2, y2 = t["bbox"]
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w/2
            y_center = y1 + h/2

            detection = Detection2D()
            detection.header = msg.header

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(self.class_mapping.get(t["class_id"], "unknown"))
            hypothesis.hypothesis.score = 1.0
            detection.results.append(hypothesis)

            if hypothesis.hypothesis.class_id == "ball":
                ball += 1
            elif hypothesis.hypothesis.class_id == "drone":
                drone += 1

            detection.bbox.center.position.x = float(x_center)
            detection.bbox.center.position.y = float(y_center)
            detection.bbox.size_x = float(w)
            detection.bbox.size_y = float(h)

            detections_msg.detections.append(detection)


        self.get_logger().info(f"Published ball: {ball}, drone: {drone}")
        self.pub_detection.publish(detections_msg)

        self.prev_gray = gray.copy()



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
