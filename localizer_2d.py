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

        # self.subscription = self.create_subscription(
        #     Image,
        #     '/camera/image',
        #     self.image_callback,
        #     qos_profile)

        self.pub_detection = self.create_publisher(Detection2DArray, 'detections_bb', qos_profile_sensor_data)

        
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


    def image_callback(self, msg):
        try:
            current_frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.total_frame_count += 1
        if ((self.image_save) and (self.saved_image_count < self.max_images) and (self.total_frame_count % self.save_interval == 0)):
            file_name = f"image_{self.saved_image_count + 1:03d}.png"
            save_path = os.path.join(self.save_dir, file_name)
            cv2.imwrite(save_path, current_frame)
            self.saved_image_count += 1


        results = self.model.track(current_frame, classes=key(class_mapping), conf=0.1, persist=True, verbose=False)

        ball = 0
        drone = 0

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # YOLO 포맷: xywh (center_x, center_y, width, height) - 정규화 안 된 픽셀 좌표
                # xyxy 등도 있지만, ROS vision_msgs는 Center 좌표를 선호하므로 xywh가 편함
                x_center, y_center, w, h = box.xywh[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = -1
                if box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())

                if (self.class_mapping[cls_id] == "ball"):
                    ball += 1
                elif (self.class_mapping[cls_id] == "drone"):
                    drone += 1

                # 개별 탐지 객체 생성
                detection = Detection2D()
                detection.header = msg.header
                
                # 가설(Hypothesis) 정보 채우기 (클래스 ID, 점수)
                hypothesis = ObjectHypothesisWithPose()

                hypothesis.hypothesis.class_id = str(self.class_mapping[cls_id]) # 문자열 ID
                hypothesis.hypothesis.score = conf
                detection.results.append(hypothesis)

                # Bounding Box 정보 채우기 (Center X, Y, Size X, Y)
                detection.bbox.center.position.x = float(x_center)
                detection.bbox.center.position.y = float(y_center)
                detection.bbox.size_x = float(w)
                detection.bbox.size_y = float(h)

                # 리스트에 추가
                detections_msg.detections.append(detection)
            
            # 로그 출력 (선택 사항)
            # self.get_logger().info(f"Published ball: {ball}, drone: {drone}", throttle_duration_sec=1.0)

        # 3. Publish
        self.get_logger().info(f"Published ball: {ball}, drone: {drone}")
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
