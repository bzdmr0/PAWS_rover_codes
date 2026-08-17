# dog_detector.py

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
import json
import ast
import time
import threading
from pathlib import Path

CONFIDENCE_THRESHOLD = 0.65
ID_RESET_TIME = 50.0

def resolve_model_path():
    env_model = os.getenv("DOG_MODEL_PATH", "").strip()
    candidates = []

    if env_model:
        candidates.append(Path(env_model))

    module_dir = Path(__file__).resolve().parent
    candidates.extend([
        module_dir / "best.engine",
        module_dir / "best.pt",
        Path.cwd() / "best.engine",
        Path.cwd() / "best.pt",
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Model file not found. Set DOG_MODEL_PATH or place best.pt/best.engine in package directory. "
        f"Searched:\n{searched}"
    )

class DogDetectorNode(Node):
    def __init__(self):
        super().__init__("DogDetectorNode")

        self.declare_parameter("camera_device_index", 0)
        self.declare_parameter("camera_width", 1280)
        self.declare_parameter("camera_height", 720)
        self.declare_parameter("camera_fps", 15)
        self.declare_parameter("use_jetson_csi", True)
        self.declare_parameter("camera_sensor_id", 0)
        self.declare_parameter("save_recording", False)
        self.declare_parameter("recording_path", "dog_detector_test.mp4")
        self.declare_parameter("recording_fps", 15.0)

        self.dog_info_pub = self.create_publisher(String, "/dog_info", 10)
        self.gps_sub = self.create_subscription(String, "/current_info", self.gps_callback, 10)

        model_path = resolve_model_path()
        self.get_logger().info(f"Using YOLO model: {model_path}")
        self.model = YOLO(model_path, task="detect")

        self.camera_device_index = int(self.get_parameter("camera_device_index").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.use_jetson_csi = bool(self.get_parameter("use_jetson_csi").value)
        self.camera_sensor_id = int(self.get_parameter("camera_sensor_id").value)
        self.save_recording = bool(self.get_parameter("save_recording").value)
        self.recording_path = str(self.get_parameter("recording_path").value).strip()
        self.recording_fps = float(self.get_parameter("recording_fps").value)

        self.cap = self._create_capture()
        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened. Check camera connection and camera parameters.")

        self.video_writer = None
        if self.save_recording:
            if not self.recording_path:
                timestamp = int(time.time())
                self.recording_path = f"dog_detector_test_{timestamp}.mp4"
            self.get_logger().info(f"Recording enabled. Output: {self.recording_path}")

        self.dog_history = {}
        self.id_reset_time = ID_RESET_TIME
        self.dog_distance = None
        self.current_lat = None
        self.current_lon = None

        self.running = True
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.start()

    def _create_capture(self):
        if self.use_jetson_csi:
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id={self.camera_sensor_id} ! "
                f"video/x-raw(memory:NVMM), width=(int){self.camera_width}, height=(int){self.camera_height}, "
                f"framerate=(fraction){self.camera_fps}/1 ! "
                "nvvidconv ! video/x-raw, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true"
            )
            self.get_logger().info(f"Opening Jetson CSI camera (sensor-id={self.camera_sensor_id})")
            return cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        self.get_logger().info(f"Opening V4L2 camera device index: {self.camera_device_index}")
        cap = cv2.VideoCapture(self.camera_device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        return cap

    def gps_callback(self, msg):
        try:
            try:
                data = json.loads(msg.data)
            except Exception:
                data = ast.literal_eval(msg.data)
            self.current_lat = data["lat"]
            self.current_lon = data["lon"]
        except Exception as e:
            self.get_logger().error(f"Failed to parse GPS data: {e}")

    def camera_loop(self):
        self.get_logger().info("Camera thread started")

        while self.running and self.cap.isOpened():
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("Failed to capture image")
                time.sleep(0.1)
                continue

            current_time = time.time()
            results = self.model.track(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                persist=True,
                stream=True
            )

            frame_to_record = frame
            for result in results:
                annotated_frame = result.plot()
                frame_to_record = annotated_frame
                if result.boxes is not None:
                    if len(result.boxes) == 0:
                        continue

                    if result.boxes.id is None:
                        self.get_logger().warn(
                            "Dog detected in frame but tracking IDs are not available yet; skipping /dog_info publish.",
                            throttle_duration_sec=5.0,
                        )
                        continue

                    track_ids = result.boxes.id.int().cpu().tolist()
                    for track_id in track_ids:
                        should_report = False
                        if track_id not in self.dog_history:
                            should_report = True
                        else:
                            last_seen = self.dog_history[track_id]
                            if time.time() - last_seen > self.id_reset_time:
                                should_report = True

                        if should_report:
                            if self.report_detection(track_id):
                                self.dog_history[track_id] = current_time

            if self.save_recording:
                self._write_frame(frame_to_record)

            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.get_logger().info(f"Processing time: {elapsed:.3f}s, FPS: {fps:.1f}", 
                                   throttle_duration_sec=2.0)

    def _write_frame(self, frame):
        if frame is None:
            return

        if self.video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.recording_path,
                fourcc,
                self.recording_fps,
                (width, height)
            )
            if not self.video_writer.isOpened():
                self.get_logger().error(f"Failed to open video writer: {self.recording_path}")
                self.video_writer = None
                self.save_recording = False
                return

        self.video_writer.write(frame)
            
    def report_detection(self, track_id):
        if self.current_lat is None or self.current_lon is None:
            self.get_logger().warn(
                f"Dog detected with ID {track_id} but GPS data is missing; skipping /dog_info publish.",
                throttle_duration_sec=5.0,
            )
            return False

        msg = String()
        detection_info = {
            "dog_id": track_id,
            "lat": self.current_lat,
            "lon": self.current_lon,
            "timestamp": self.dog_history[track_id]
        }
        msg.data = json.dumps(detection_info)
        self.dog_info_pub.publish(msg)
        self.get_logger().info(f"Dog detected with ID: {track_id}")
        return True

    def shutdown(self):
        self.running = False
        if self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2.0)
        if self.video_writer is not None:
            self.video_writer.release()
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = DogDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()