# server_bridge.py

import os
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import String
import requests
import json
import ast
import time
from datetime import datetime
from rover_interface.srv import ServerCommands, MissionControl

class ServerBridgeNode(Node):
    def __init__(self):
        super().__init__('server_bridge')

        default_base_url = os.getenv('ROVER_SERVER_BASE_URL', 'http://37.140.242.36').strip()
        self.declare_parameter('base_url', default_base_url)
        self.base_url = str(self.get_parameter('base_url').value).strip().rstrip('/')
        self.declare_parameter('test_mode', False)
        self.declare_parameter(
            'test_path_json',
            '(35.248049,33.022945),(35.248017,33.023042)',
            ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter('test_send_once', True)
        self.declare_parameter('service_request_timeout_sec', 8.0)
        self.test_sent = False
        self.last_sent_path_signature = None
        self.current_path_id = None
        self.pending_request_started_monotonic = None
        self.last_mission_state = None
        self.last_control_command = None

        self.create_subscription(String, '/current_info', self.location_callback, 10)
        self.create_subscription(String, '/dog_info', self.dog_callback, 10)
        self.create_subscription(String, '/mission_state', self.mission_state_callback, 10)

        self.cli = self.create_client(ServerCommands, '/server_commands')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.pending_future = None

        self.control_cli = self.create_client(MissionControl, '/mission_control')
        while not self.control_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('mission_control service not available, waiting again...')
        self.pending_control_future = None

        self.create_timer(2.0, self.check_path)

    def location_callback(self, msg):
        try:
            data = json.loads(msg.data)
            payload = {
                'latitude': data['lat'],
                'longitude': data['lon'],
                'cpu_temperature': data['cpu_temperature']
            }

            test_mode = bool(self.get_parameter('test_mode').value)
            if test_mode:
                self.get_logger().info(f"[TEST MODE] Would send location to server: {payload}")
            else:
                requests.post(f"{self.base_url}/api/rover/location", json=payload, timeout=5)
                self.get_logger().info("Location sent to server")
        except Exception as e:
            self.get_logger().error(f"Location send error: {e}")

    def dog_callback(self, msg):
        try:
            data = json.loads(msg.data)
            ts = data.get('timestamp')
            if ts is not None:
                formatted_ts = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            else:
                formatted_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            payload = {
                'coordinate': f"{data['lat']}, {data['lon']}",
                'timestamp': formatted_ts
            }

            test_mode = bool(self.get_parameter('test_mode').value)
            if test_mode:
                self.get_logger().info(f"[TEST MODE] Would send dog detection to server: {payload}")
            else:
                requests.post(f"{self.base_url}/api/dog_detected", json=payload, timeout=5)
                self.get_logger().info("Dog detection sent to server")
        except Exception as e:
            self.get_logger().error(f"Dog detection send error: {e}")

    def check_path(self):
        try:
            test_mode = bool(self.get_parameter('test_mode').value)
            if test_mode:
                test_path_json = str(self.get_parameter('test_path_json').value)
                self.get_logger().info(f"[TEST MODE] Checking path with test_path_json: {test_path_json}")
                send_once = bool(self.get_parameter('test_send_once').value)

                if send_once and self.test_sent:
                    return

                try:
                    path = json.loads(test_path_json)
                except json.JSONDecodeError as e:
                    try:
                        path = list(ast.literal_eval(f"[{test_path_json}]"))
                        self.get_logger().info(f"[TEST MODE] Parsed test_path_json as tuple-style: {path}")
                    except (ValueError, SyntaxError) as parse_err:
                        self.get_logger().error(
                            f"[TEST MODE] Invalid test_path_json: {e}; tuple parse failed: {parse_err}"
                        )
                        return

                if self._send_path_request(path, source='TEST MODE'):
                    self.test_sent = True
                return

            self.get_logger().warn("Checking for new path assignment from server...")
            response = requests.get(f"{self.base_url}/api/rover/get_selected_path", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                path_id = data.get('path_id')
                self.get_logger().info(f"[DEBUG] Server response: path_id={path_id}, current_path_id={self.current_path_id}")
                if path_id is None:
                    if self.current_path_id is not None:
                        self.get_logger().warn(f"[DEBUG] Path cancelled: had path_id={self.current_path_id}, sending ABORT...")
                        if self._send_control_command('ABORT', source='SERVER'):
                            self.get_logger().warn('No active path on server; sent ABORT for current mission.')
                        return

                    self.get_logger().info("[DEBUG] path_id is None and no current_path_id, clearing last_control_command")
                    self.last_control_command = None
                    return

                command = str(data.get('command', '')).strip().upper()
                if command in ("STOP", "RUN"):
                    self.get_logger().info(f"Received control command from server: {command}")
                    self._send_control_command(command, source='SERVER')

                coordinates = data.get('coordinates', [])
                self._send_path_request(coordinates, source='SERVER', path_id=path_id)
            else:
                self.get_logger().warn(f"Path endpoint returned status {response.status_code}")
        except Exception as e:
            self.get_logger().error(f"Path check error: {e}")

    def _send_path_request(self, path, source='SERVER', path_id=None):
        if not isinstance(path, list) or len(path) == 0:
            self.get_logger().error(f"[{source}] Invalid path format. Expected non-empty list, got: {path}")
            return False

        path_json = json.dumps(path, separators=(',', ':'))

        if path_json == self.last_sent_path_signature and (path_id is None or path_id == self.current_path_id):
            return False

        if self.pending_future is not None and not self.pending_future.done():
            request_timeout_sec = float(self.get_parameter('service_request_timeout_sec').value)
            if self.pending_request_started_monotonic is not None:
                if time.monotonic() - self.pending_request_started_monotonic > request_timeout_sec:
                    self.get_logger().warn(
                        f'[{source}] Previous path request timed out; resetting pending request state.'
                    )
                    self.pending_future = None
                    self.pending_request_started_monotonic = None
                else:
                    self.get_logger().info(f'[{source}] Previous path request still pending; skipping this cycle.')
                    return False
            else:
                self.get_logger().info(f'[{source}] Previous path request still pending; skipping this cycle.')
                return False

        req = ServerCommands.Request()
        req.path_json = path_json
        try:
            self.pending_future = self.cli.call_async(req)
        except Exception as e:
            self.pending_future = None
            self.pending_request_started_monotonic = None
            self.get_logger().error(f"[{source}] Service call start failed: {e}")
            return False

        self.pending_request_started_monotonic = time.monotonic()
        self.pending_future.add_done_callback(
            lambda future, sent_path_json=path_json, sent_path_id=path_id, sent_source=source:
            self._on_path_response(future, sent_path_json, sent_path_id, sent_source)
        )
        self.get_logger().info(f"[{source}] Normalized path JSON: {path}")
        self.get_logger().info(f"[{source}] Sent waypoint list with {len(path)} waypoints to /server_commands")
        return True

    def _send_control_command(self, command, source='SERVER'):
        if not command:
            return False

        if command != 'ABORT' and command == self.last_control_command:
            return False

        if self.pending_control_future is not None and not self.pending_control_future.done():
            return False

        req = MissionControl.Request()
        req.command = command
        try:
            self.pending_control_future = self.control_cli.call_async(req)
        except Exception as e:
            self.pending_control_future = None
            self.get_logger().error(f"[{source}] Mission control call failed: {e}")
            return False

        self.pending_control_future.add_done_callback(
            lambda future, sent_command=command, sent_source=source:
            self._on_control_response(future, sent_command, sent_source)
        )
        self.get_logger().info(f"[{source}] Sent mission control command: {command}")
        return True

    def _on_control_response(self, future, sent_command, sent_source):
        self.pending_control_future = None
        try:
            response = future.result()
            if response.success:
                self.last_control_command = sent_command
                self.get_logger().info(f"[{sent_source}] Mission control accepted: {response.message}")
            else:
                if self.last_control_command == sent_command:
                    self.last_control_command = None
                self.get_logger().warn(f"[{sent_source}] Mission control rejected: {response.message}")
        except Exception as e:
            if self.last_control_command == sent_command:
                self.last_control_command = None
            self.get_logger().error(f"[{sent_source}] Mission control service call failed: {e}")

    def _on_path_response(self, future, sent_path_json, sent_path_id, sent_source):
        self.pending_future = None
        self.pending_request_started_monotonic = None
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Path update accepted: {response.message}")
                self.last_sent_path_signature = sent_path_json
                if sent_path_id is not None:
                    self.current_path_id = sent_path_id
            else:
                self.get_logger().warn(f"Path update rejected: {response.message}")
                if self.last_sent_path_signature == sent_path_json:
                    self.last_sent_path_signature = None
                self.get_logger().info(
                    f"[{sent_source}] Rejected path will be retried on next check_path cycle."
                )
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            if self.last_sent_path_signature == sent_path_json:
                self.last_sent_path_signature = None

    def mission_state_callback(self, msg):
        state = str(msg.data).strip()
        if state == self.last_mission_state:
            return

        self.last_mission_state = state
        self.get_logger().warn(f"Mission state update: {state}")

        if state == "COMPLETED":
            self.last_control_command = None
            self.notify_path_completed(self.current_path_id)
        elif state == "CANCELLED":
            self.last_sent_path_signature = None
            self.current_path_id = None
            self.last_control_command = None
            self.get_logger().warn("Mission cancelled. Cleared active path state.")
        elif state == "FAILED":
            self.last_sent_path_signature = None
            self.current_path_id = None
            self.last_control_command = None
            self.get_logger().warn("Mission failed. Cleared active path signature for retry.")

    def notify_path_completed(self, path_id=None):
        if path_id is None:
            path_id = self.current_path_id

        if path_id is None:
            self.get_logger().warn("notify_path_completed called but no path_id is set")
            return

        payload = {'path_id': path_id}

        test_mode = bool(self.get_parameter('test_mode').value)
        if test_mode:
            self.get_logger().info(f"[TEST MODE] Would notify server: Path #{path_id} completed")
            self.last_sent_path_signature = None
            self.current_path_id = None
            return

        try:
            response = requests.post(
                f"{self.base_url}/api/rover/path_completed",
                json=payload,
                timeout=5
            )
            if response.status_code == 200:
                self.get_logger().info(f"Server notified: Path #{path_id} completed. Rover is now available.")
                self.last_sent_path_signature = None
                self.current_path_id = None
            else:
                resp_data = response.json()
                self.get_logger().warn(
                    f"Path completion rejected: {resp_data.get('message', response.status_code)}"
                )
        except Exception as e:
            self.get_logger().error(f"Failed to notify path completion: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ServerBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()