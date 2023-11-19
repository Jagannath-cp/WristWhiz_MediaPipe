#!/usr/bin/env python3

import os
import sys
import traceback
from xarm import version
from xarm.wrapper import XArmAPI

import cv2
import mediapipe as mp
import time
import numpy as np
import pyrealsense2 as rs

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


class RobotMain(object):
    """Robot Main Class"""

    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code,
                                                                                                 self._arm.connected,
                                                                                                 self._arm.state,
                                                                                                 self._arm.error_code,
                                                                                                 ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1],
                                       ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    # Robot Main Run
    def run(self):
        self._arm.set_position(293.3, 10.4, 484.7, -179.9, 1.4, 2.3, speed=self._tcp_speed,
                               mvacc=self._tcp_acc, radius=0.0, wait=False) #starting position
        try:
            cx = 0
            cy = 0
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            pipeline.start(config)

            # Create alignment object
            align_to = rs.stream.color
            align = rs.align(align_to)

            mpHands = mp.solutions.hands
            hands = mpHands.Hands(static_image_mode=False,
                                  max_num_hands=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
            mpDraw = mp.solutions.drawing_utils

            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                # Convert images to numpy arrays
                img = np.asanyarray(color_frame.get_data())

                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        for id, lm in enumerate(handLms.landmark):
                            h, w, c = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            if id == 8:
                                print("X and Y coordinates at first landmark", cx, cy)
                                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                                # Linear Motion
                                self._tcp_speed = 200
                                self._tcp_acc = 5000

                                depth = depth_frame.get_distance(cx, cy)
                                dx, dy, dz = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
                                print(dx, dy, dz)
                                if not self.is_alive:
                                    break
                                current_position = self._arm.get_position()
                                print(current_position)
                                print("init", current_position)
                                current_position[1][0] = 293.3
                                if (235 < (current_position[1][2] - (1000 * dy)) < 690) and (-500 < current_position[1][1] - (1000 * dx) < 500):
                                    code = self._arm.set_position(
                                        *[current_position[1][0], current_position[1][1] - (1000 * dx),
                                          current_position[1][2] - (1000 * dy),
                                          current_position[1][3], current_position[1][4], current_position[1][5]],
                                        speed=self._tcp_speed,
                                        mvacc=self._tcp_acc, radius=0.0, wait=False)
                                    time.sleep(0.5)
                                else:
                                    time.sleep(5)
                                    self._arm.set_position(293.3, 10.4, 484.7, -179.9, 1.4, 2.3, speed=self._tcp_speed,
                                                           mvacc=self._tcp_acc, radius=0.0, wait=False) # starting position

                cv2.imshow("Image", img)
                cv2.waitKey(500)

        except Exception as e:
            self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)

            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)


if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.198', baud_checkset=False)
    robot_main = RobotMain(arm)
    robot_main.run()
