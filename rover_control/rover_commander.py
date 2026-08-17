# rover_commander.py

import collections
import collections.abc
import glob
import os

if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import time
import math
import threading
import ast
from json import dumps, loads
import rclpy
from rclpy.node import Node
from rover_interface.srv import ServerCommands, MissionControl
from std_msgs.msg import String


DISTANCE_THRESHOLD = 3.0  
AUTO_ARM_TIMEOUT_SEC = 10.0
GUIDED_TIMEOUT_SEC = 20.0
HEALTH_CHECK_TIMEOUT_SEC = 45.0
RTL_TIMEOUT_SEC = 300.0
GPS_FIX_MIN = 3
MAX_HDOP = 2.5
MIN_SATELLITES = 6


class RoverCommanderNode(Node):
    def __init__(self):
        super().__init__('rover_commander')
        self.vehicle = None
        self.path = None
        self._mission_thread = None
        self._current_info_thread = None
        self._current_info_stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self.target_wp = None
        self.state = "HOLD"

        self.declare_parameter('connection_string', '')
        self.declare_parameter('baud_rate', 57600)
        self.declare_parameter('connect_timeout_sec', 30.0)
        self.declare_parameter('connect_retry_count', 3)
        self.declare_parameter('connect_retry_backoff_sec', 2.0)
        self.declare_parameter('waypoint_reach_threshold_m', DISTANCE_THRESHOLD)
        self.declare_parameter('waypoint_retry_count', 2)
        self.declare_parameter('heartbeat_timeout_sec', 10.0)
        self.declare_parameter('fallback_mode', 'HOLD')
        self.declare_parameter('rtl_timeout_sec', RTL_TIMEOUT_SEC)

        self.server_srv = self.create_service(ServerCommands, "/server_commands", self.from_server_callback)
        self.mission_control_srv = self.create_service(MissionControl, "/mission_control", self.mission_control_callback)

        self.current_info = self.create_publisher(String, "/current_info", 10)
        self.mission_state_pub = self.create_publisher(String, "/mission_state", 10)

    def from_server_callback(self, request, response):
        try:
            if self.state != "HOLD" or (self._mission_thread is not None and self._mission_thread.is_alive()):
                response.success = False
                response.message = f"Mission is busy. Current state: {self.state}"
                return response

            if not request.path_json:
                response.success = False
                response.message = "Empty path_json received."
                self.get_logger().error(response.message)
                return response

            raw_path = self._parse_path_payload(request.path_json)
            received_path = self._normalize_path(raw_path)
            self.get_logger().info(f"Received direct path with {len(received_path)} waypoints from server.")

            self.path = received_path
            self.target_wp = 0
            self._pause_event.clear()
            self._cancel_event.clear()
            self._set_state("STARTING")

            self._mission_thread = threading.Thread(target=self.patrol, daemon=True)
            self._mission_thread.start()

            response.success = True
            response.message = f"Path set successfully with {len(self.path)} waypoints."
        except Exception as e:
            response.success = False
            response.message = f"Error setting path: {str(e)}"
            self.get_logger().error(f"Failed to parse path data: {e}")
        
        return response

    def mission_control_callback(self, request, response):
        command = str(request.command).strip().upper()
        if command == "STOP":
            success, message = self._pause_mission()
        elif command == "RUN":
            success, message = self._resume_mission()
        elif command == "ABORT":
            success, message = self._abort_mission()
        else:
            success = False
            message = f"Unknown command: {request.command}"

        response.success = success
        response.message = message
        return response

    def _pause_mission(self):
        if self._mission_thread is None or not self._mission_thread.is_alive():
            return False, "No active mission to pause."

        self._pause_event.set()
        if self.vehicle is not None:
            if not self._set_mode("HOLD", GUIDED_TIMEOUT_SEC):
                self.get_logger().warn("Pause requested but failed to switch to HOLD mode.")
        self._set_state("PAUSED")
        return True, "Pause requested."

    def _resume_mission(self):
        if self._mission_thread is None or not self._mission_thread.is_alive():
            return False, "No active mission to resume."

        if not self._pause_event.is_set():
            return False, "Mission is not paused."

        self._pause_event.clear()
        self._set_state("RUNNING")
        return True, "Resume requested."

    def _abort_mission(self):
        if self._mission_thread is None or not self._mission_thread.is_alive():
            return False, "No active mission to abort."

        self._cancel_event.set()
        self._pause_event.clear()
        if self.vehicle is not None:
            if not self._set_mode("HOLD", GUIDED_TIMEOUT_SEC):
                self.get_logger().warn("Abort requested but failed to switch to HOLD mode.")
        return True, "Abort requested."

    def _set_state(self, new_state):
        self.state = new_state
        self.get_logger().warn(f"State changed to: {new_state}")
        msg = String()
        msg.data = new_state
        self.mission_state_pub.publish(msg)

    def _parse_path_payload(self, payload):
        try:
            parsed = loads(payload)
        except Exception:
            parsed = list(ast.literal_eval(f"[{payload}]"))

        if isinstance(parsed, dict):
            parsed = parsed.get("path")

        return parsed

    def _normalize_path(self, raw_path):
        if not isinstance(raw_path, list) or len(raw_path) == 0:
            raise ValueError("Path must be a non-empty list of waypoints.")

        normalized = []
        for i, wp in enumerate(raw_path, start=1):
            if not isinstance(wp, (list, tuple)) or len(wp) < 2:
                raise ValueError(f"Waypoint {i} is invalid. Expected at least [lat, lon].")
            lat = float(wp[0])
            lon = float(wp[1])
            alt = float(wp[2]) if len(wp) > 2 else 0
            normalized.append((lat, lon, alt))

        return normalized

    def connectMyCopter(self):
        connection_string = str(self.get_parameter('connection_string').value).strip()
        if not connection_string:
            raise ValueError("Missing connection string. Use '-- --connect <endpoint>' or set ROS param 'connection_string'.")

        baud_rate = int(self.get_parameter('baud_rate').value)
        connect_timeout = float(self.get_parameter('connect_timeout_sec').value)
        retry_count = int(self.get_parameter('connect_retry_count').value)
        retry_backoff = float(self.get_parameter('connect_retry_backoff_sec').value)

        last_error = None
        for attempt in range(1, retry_count + 1):
            if self._is_cancelled():
                return None
            try:
                self.get_logger().info(f"Connecting to vehicle (attempt {attempt}/{retry_count}) on {connection_string}.")
                self.vehicle = connect(
                    connection_string,
                    baud=baud_rate,
                    wait_ready=True,
                    timeout=connect_timeout
                )
                self.get_logger().info("Vehicle connection established.")
                return self.vehicle
            except Exception as e:
                last_error = e
                self.get_logger().error(f"Vehicle connection attempt failed: {e}")
                if attempt < retry_count:
                    if self._is_cancelled():
                        return None
                    time.sleep(retry_backoff)

        raise RuntimeError(f"Failed to connect to vehicle after {retry_count} attempts: {last_error}")

    def _wait_until(self, condition_fn, timeout_sec, wait_reason):
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            if self._is_cancelled():
                return False
            try:
                if condition_fn():
                    return True
            except Exception as e:
                self.get_logger().warn(f"Condition check error for {wait_reason}: {e}")
            time.sleep(1)

        self.get_logger().error(f"Timeout while waiting for: {wait_reason}")
        return False

    def _set_mode(self, mode_name, timeout_sec):
        try:
            self.vehicle.mode = VehicleMode(mode_name)
        except APIException as e:
            self.get_logger().error(f"Failed to set mode {mode_name}: {e}")
            return False

        return self._wait_until(lambda: self.vehicle.mode.name == mode_name, timeout_sec, f"mode {mode_name}")

    def _check_prearm_health(self):
        start = time.monotonic()
        while time.monotonic() - start < HEALTH_CHECK_TIMEOUT_SEC:
            if self._is_cancelled():
                return False
            gps = getattr(self.vehicle, 'gps_0', None)
            fix_type = int(getattr(gps, 'fix_type', 0) or 0)
            hdop_raw = float(getattr(gps, 'eph', 99.9) or 99.9)
            hdop = hdop_raw / 100.0 if hdop_raw > 10.0 else hdop_raw
            sats = int(getattr(gps, 'satellites_visible', 0) or 0)
            ekf_ok = bool(getattr(self.vehicle, 'ekf_ok', False))

            home_location = getattr(self.vehicle, 'home_location', None)
            if home_location is None:
                try:
                    cmds = self.vehicle.commands
                    cmds.download()
                    cmds.wait_ready()
                    home_location = getattr(self.vehicle, 'home_location', None)
                    self.get_logger().info(f"Home location after retry: {home_location}")
                except Exception:
                    pass

            home_set = home_location is not None

            if fix_type >= GPS_FIX_MIN and ekf_ok and hdop <= MAX_HDOP and sats >= MIN_SATELLITES and home_set:
                return True

            self.get_logger().warn(
                f"Pre-arm waiting: fix={fix_type}/{GPS_FIX_MIN}, ekf_ok={ekf_ok}, hdop={hdop:.2f}/{MAX_HDOP}, "
                f"sats={sats}/{MIN_SATELLITES}, home_set={home_set}",
                throttle_duration_sec=5.0
            )
            time.sleep(1)

        return False

    def _connection_healthy(self):
        if self.vehicle is None:
            return False

        heartbeat_timeout = float(self.get_parameter('heartbeat_timeout_sec').value)
        last_heartbeat = getattr(self.vehicle, 'last_heartbeat', None)
        if last_heartbeat is not None and float(last_heartbeat) > heartbeat_timeout:
            self.get_logger().error(f"Connection unhealthy: last heartbeat {last_heartbeat:.2f}s exceeds {heartbeat_timeout:.2f}s")
            return False

        return True

    def arm(self):
        while self.vehicle.is_armable is not True:
            if self._is_cancelled():
                return False
            self.get_logger().info("Waiting for vehicle to become armable...")
            time.sleep(1)

        if not self._check_prearm_health():
            self.get_logger().error("Pre-arm health check failed.")
            return False

        if not self._set_mode("GUIDED", GUIDED_TIMEOUT_SEC):
            fallback_mode = str(self.get_parameter('fallback_mode').value)
            self._set_mode(fallback_mode, GUIDED_TIMEOUT_SEC)
            return False

        self.vehicle.armed = True
        while self.vehicle.armed is not True:
            if self._is_cancelled():
                return False
            self.get_logger().info("Waiting for vehicle to arm...")
            time.sleep(1)

        self.get_logger().info("Vehicle armed successfully.")

        return True

    def _wait_for_manual_arm(self, timeout_sec):
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            if self._is_cancelled():
                return False
            if self.vehicle is not None and self.vehicle.armed:
                return True
            time.sleep(0.5)
        return False

    def _ensure_armed_with_auto_arm(self):
        if self.vehicle is None:
            return False

        if self.vehicle.armed:
            return self._set_mode("GUIDED", GUIDED_TIMEOUT_SEC)

        if self._wait_for_manual_arm(AUTO_ARM_TIMEOUT_SEC):
            self.get_logger().info("Vehicle armed manually before auto-arm timeout.")
            return self._set_mode("GUIDED", GUIDED_TIMEOUT_SEC)

        self.get_logger().warn("Auto-arming after 10 seconds without manual arm.")
        return self.arm()

    def get_jetson_temperature(self):
        try:
            for tz in glob.glob('/sys/class/thermal/thermal_zone*'):
                type_path = os.path.join(tz, 'type')
                temp_path = os.path.join(tz, 'temp')
                try:
                    with open(type_path, 'r') as f:
                        type_name = f.read().strip()
                    if type_name != 'tj-thermal':
                        continue
                    with open(temp_path, 'r') as f:
                        return int(f.read().strip()) / 1000.0
                except Exception:
                    continue

            fallback_path = '/sys/devices/virtual/thermal/tz-soc-thermal/temp'
            if os.path.exists(fallback_path):
                with open(fallback_path, 'r') as f:
                    return int(f.read().strip()) / 1000.0

            self.get_logger().warn('tj-thermal sensor not found.')
            return None
        except Exception as e:
            self.get_logger().warn(f"Failed to read temperature: {e}")
            return None

    def get_distance_meters(self,targetLocation,currentLocation):
        dLat=targetLocation.lat - currentLocation.lat
        dLon=targetLocation.lon - currentLocation.lon
        
        return math.sqrt((dLon*dLon)+(dLat*dLat))*1.113195e5

    def publish_current_info(self):
        if self.vehicle is None or self.vehicle.location is None:
            self.get_logger().warn("Vehicle or vehicle location is not available. Skipping current info publish.")
            return

        frame = self.vehicle.location.global_relative_frame
        if frame is None or frame.lat is None or frame.lon is None:
            self.get_logger().warn("Vehicle location data is incomplete. Skipping current info publish.")
            return

        info_msg = String()
        temp = self.get_jetson_temperature()
        info_data = {
            "lat": frame.lat,
            "lon": frame.lon,
            "target_wp": self.target_wp,
            "state": self.state,
            "cpu_temperature": temp,
            "timestamp": self.get_clock().now().to_msg().sec
        }
        info_msg.data = dumps(info_data)
        self.current_info.publish(info_msg)

    def goto(self,targetLocation, wp_id):
        threshold_m = float(self.get_parameter('waypoint_reach_threshold_m').value)
        retry_count = int(self.get_parameter('waypoint_retry_count').value)

        for attempt in range(1, retry_count + 1):
            if self._cancel_event.is_set():
                self.get_logger().warn(f"[goto] Waypoint {wp_id}: Cancel requested during attempt {attempt}")
                return "CANCELLED"
            try:
                self.get_logger().info(f"[goto] Waypoint {wp_id} (attempt {attempt}): Calling simple_goto...")
                self.vehicle.simple_goto(targetLocation)
                self.get_logger().info(f"[goto] Waypoint {wp_id}: simple_goto succeeded, mode={self.vehicle.mode.name}")
            except APIException as e:
                self.get_logger().error(f"simple_goto failed for waypoint {wp_id}: {e}")
                continue

            while self.vehicle.mode.name == "GUIDED":
                if self._cancel_event.is_set():
                    self.get_logger().warn(f"[goto] Waypoint {wp_id}: Cancel requested in GUIDED loop")
                    return "CANCELLED"
                if self._pause_event.is_set():
                    self.get_logger().info(f"[goto] Waypoint {wp_id}: Pause requested")
                    return "PAUSED"
                if not self._connection_healthy():
                    self.get_logger().error(f"[goto] Waypoint {wp_id}: Connection unhealthy")
                    return False

                frame = getattr(self.vehicle.location, 'global_relative_frame', None)
                if frame is None or frame.lat is None or frame.lon is None:
                    self.get_logger().warn(f"[goto] Waypoint {wp_id}: No GPS location yet")
                    time.sleep(1)
                    continue

                currentDistance = self.get_distance_meters(targetLocation, frame)
                self.get_logger().debug(f"[goto] Waypoint {wp_id}: Distance={currentDistance:.2f}m, Threshold={threshold_m}m")

                if currentDistance <= threshold_m:
                    self.get_logger().info(f"*****Reached target waypoint {wp_id}. *******")
                    time.sleep(1)
                    return True

                time.sleep(1)

            self.get_logger().warn(f"[goto] Waypoint {wp_id}: Exited GUIDED mode (current mode: {self.vehicle.mode.name})")
            if self._pause_event.is_set():
                return "PAUSED"

        self.get_logger().error(f"Waypoint {wp_id} failed after {retry_count} attempts.")
        return False

    def _wait_for_rtl_completion(self):
        home_threshold_m = float(self.get_parameter('waypoint_reach_threshold_m').value)
        rtl_timeout_sec = float(self.get_parameter('rtl_timeout_sec').value)
        hold_mode_timeout_sec = GUIDED_TIMEOUT_SEC

        home_location = getattr(self.vehicle, 'home_location', None)
        if home_location is None:
            try:
                cmds = self.vehicle.commands
                cmds.download()
                cmds.wait_ready()
                home_location = getattr(self.vehicle, 'home_location', None)
            except Exception:
                pass

        if home_location is None:
            self.get_logger().error("RTL completion check failed: home location is not available.")
            return False

        start = time.monotonic()
        while True:
            if not self._connection_healthy():
                return False

            if time.monotonic() - start > rtl_timeout_sec:
                self.get_logger().error(f"RTL completion timed out after {rtl_timeout_sec:.1f}s.")
                return False

            frame = getattr(self.vehicle.location, 'global_relative_frame', None)
            if frame is not None and frame.lat is not None and frame.lon is not None:
                home_distance = self.get_distance_meters(home_location, frame)
                if home_distance <= home_threshold_m:
                    self.get_logger().info(
                        f"Home threshold reached ({home_distance:.2f}m <= {home_threshold_m:.2f}m). Switching to HOLD."
                    )
                    if not self._set_mode("HOLD", hold_mode_timeout_sec):
                        self.get_logger().error("Failed to switch to HOLD after reaching home threshold in RTL.")
                        return False
                    self.get_logger().info("RTL completed successfully and vehicle switched to HOLD.")
                    return True

            time.sleep(1)

    def _current_info_loop(self):
        while not self._current_info_stop_event.is_set():
            if self.vehicle is not None:
                self.publish_current_info()
            time.sleep(1)

    def _start_current_info_stream(self):
        if self._current_info_thread is not None and self._current_info_thread.is_alive():
            return

        self._current_info_stop_event.clear()
        self._current_info_thread = threading.Thread(target=self._current_info_loop, daemon=True)
        self._current_info_thread.start()

    def _stop_current_info_stream(self):
        self._current_info_stop_event.set()
        if self._current_info_thread is not None and self._current_info_thread.is_alive():
            self._current_info_thread.join(timeout=2.0)

    def _cleanup_mission_context(self):
        self._stop_current_info_stream()
        self._pause_event.clear()
        self._cancel_event.clear()
        try:
            if self.vehicle is not None:
                self.vehicle.close()
        except Exception as e:
            self.get_logger().warn(f"Vehicle close failed during cleanup: {e}")
        self.vehicle = None
        self.path = None
        self.target_wp = None
        self._mission_thread = None

    def patrol(self):
        if self.path is None:
            self.get_logger().warn("Path is not set. Mission will not start.")
            self._set_state("HOLD")
            return

        try:
            self._set_state("CONNECTING")
            self.vehicle = self.connectMyCopter()
            if self.vehicle is None and self._is_cancelled():
                self._set_state("CANCELLED")
                return
            self._start_current_info_stream()

            if self._cancel_event.is_set():
                self._set_state("CANCELLED")
                return

            pause_result = self._wait_while_paused()
            if pause_result == "CANCELLED":
                self._set_state("CANCELLED")
                return
            if not pause_result:
                self._set_state("FAILED")
                return

            self._set_state("HEALTH_CHECK")
            if not self._ensure_armed_with_auto_arm():
                if self._is_cancelled():
                    self._set_state("CANCELLED")
                    return
                self._set_state("FAILED")
                return

            if self._cancel_event.is_set():
                self._set_state("CANCELLED")
                return

            self._set_state("RUNNING")
            for (i, wp) in enumerate(self.path, start=1):
                self.target_wp = i
                alt = float(wp[2]) if len(wp) > 2 else 10.0
                while True:
                    if self._cancel_event.is_set():
                        self._set_state("CANCELLED")
                        return
                    pause_result = self._wait_while_paused()
                    if pause_result == "CANCELLED":
                        self._set_state("CANCELLED")
                        return
                    if not pause_result:
                        self._set_state("FAILED")
                        return
                    result = self.goto(LocationGlobalRelative(float(wp[0]), float(wp[1]), alt), i)
                    if result is True:
                        break
                    if result == "CANCELLED":
                        self._set_state("CANCELLED")
                        return
                    if result == "PAUSED":
                        continue
                    self._set_state("FAILED")
                    fallback_mode = str(self.get_parameter('fallback_mode').value)
                    self._set_mode(fallback_mode, GUIDED_TIMEOUT_SEC)
                    return

            if not self._set_mode("HOLD", GUIDED_TIMEOUT_SEC):
                self._set_state("FAILED")
                fallback_mode = str(self.get_parameter('fallback_mode').value)
                self._set_mode(fallback_mode, GUIDED_TIMEOUT_SEC)
                self.get_logger().error("Mission completed but failed to switch to HOLD mode.")
                return

            self._set_state("COMPLETED")
            self.get_logger().info("Mission completed successfully. Rover is holding at final waypoint.")
        except Exception as e:
            self._set_state("FAILED")
            self.get_logger().error(f"Mission thread failed: {e}")
        finally:
            self._cleanup_mission_context()
            self._set_state("HOLD")

    def _wait_while_paused(self):
        if self._cancel_event.is_set():
            return "CANCELLED"

        if not self._pause_event.is_set():
            return True

        self._set_state("PAUSED")
        while self._pause_event.is_set():
            if self._cancel_event.is_set():
                return "CANCELLED"
            if self.vehicle is not None and not self._connection_healthy():
                return False
            time.sleep(0.5)

        if self.vehicle is None:
            return True

        if not self._set_mode("GUIDED", GUIDED_TIMEOUT_SEC):
            return False

        if not self.vehicle.armed:
            return self.arm()
        return True

    def _is_cancelled(self):
        return self._cancel_event.is_set()

    def destroy_node(self):
        self._stop_current_info_stream()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoverCommanderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()