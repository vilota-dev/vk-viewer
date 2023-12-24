#!/usr/bin/env python3

import sys
import time

import capnp
import numpy as np
import cv2
from scipy.spatial import ConvexHull

import ecal.core.core as ecal_core

capnp.add_import_hook(['thirdparty/ecal_common/src/capnp'])

import image_capnp as eCALImage
import imu_capnp as eCALImu
import tagdetection_capnp as eCALTagDetection
import odometry3d_capnp as eCALOdometry3d

from thirdparty.ecal_common.python.capnp_subscriber import CapnpSubscriber

# visualisation tools
import rerun as rr

# recording
import signal
import queue
from mcap.writer import Writer
import datetime
import threading
class ImageLogger:
    def __init__(self, topic_list) -> None:
        self.subs = []
        for topic in topic_list:
            sub = CapnpSubscriber("Image", topic)
            self.subs.append(sub)
            sub.set_callback(self.callback)

    def callback(self, topic_type, topic_name, msg, ts):
        # print(f"callback of topic {topic_name}")

        with eCALImage.Image.from_bytes(msg) as imageMsg:

            mat = self.image_msg_to_cvmat(imageMsg)

            rr.set_time_nanos("host_monotonic_time", imageMsg.header.stamp)
        
            # we have to ways to prevent all images drawn to the same screen
            rr.log(topic_name, rr.DisconnectedSpace())
            # rr.log(topic_name + "/image", rr.Pinhole(focal_length=300, width=imageMsg.width*2//3, height=imageMsg.height), timeless=True)
            
            rr.log(topic_name, rr.Image(mat[:, :mat.shape[1]*2//3]))

            # some debugging drawings
            # rr.log(topic_name + "/image", rr.Boxes2D(mins=[10,20], sizes=[20,40]))
            # rr.log(topic_name + "/tag_detection" , rr.LineStrips2D([[[100, 100], [200,300], [400, 100], [600, 200]]]))

    @staticmethod
    def image_msg_to_cvmat(imageMsg):
        if (imageMsg.encoding == "mono8"):

            mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
            mat = mat.reshape((imageMsg.height, imageMsg.width, 1))

            # cv2.imshow("mono8", mat)
            # cv2.waitKey(3)
        elif (imageMsg.encoding == "yuv420"):
            mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
            mat = mat.reshape((imageMsg.height * 3 // 2, imageMsg.width, 1))

            mat = cv2.cvtColor(mat, cv2.COLOR_YUV2BGR_IYUV)

        elif (imageMsg.encoding == "bgr8"):
            mat = np.frombuffer(imageMsg.data, dtype=np.uint8)
            mat = mat.reshape((imageMsg.height, imageMsg.width, 3))
        elif (imageMsg.encoding == "jpeg"):
            mat_jpeg = np.frombuffer(imageMsg.data, dtype=np.uint8)
            mat = cv2.imdecode(mat_jpeg, cv2.IMREAD_COLOR)
        else:
            raise RuntimeError("unknown encoding: " + imageMsg.encoding)
        
        return mat
    
class TagDetectionLogger:
    def __init__(self, topic_list) -> None:
        self.subs = []
        for topic in topic_list:
            sub = CapnpSubscriber("TagDetections", topic)
            self.subs.append(sub)
            sub.set_callback(self.callback)

        # just to make some decoupling of the disk writing and visualisation
        self.queue = queue.Queue(10)
        self.writer = None
        self.channel_ids = dict()

        self.thread = threading.Thread(target=self.writer_thread)
        self.thread.start()

    def callback(self, topic_type, topic_name, msg, ts):
        # print(f"callback of topic {topic_name}")

        # we should not clear, otherwise the whole history was cleared
        # rr.log(topic_name, rr.Clear(recursive=True))

        with eCALTagDetection.TagDetections.from_bytes(msg) as tagsMsg:
            # print("tag detection received")

            rr.set_time_nanos("host_monotonic_time", tagsMsg.header.stamp)

            # print(tagsMsg)

            # obtain image size
            if tagsMsg.image.mipMapLevels == 0:
                img_width = tagsMsg.image.width
            else:
                img_width = tagsMsg.image.width * 2 // 3
            image_height = tagsMsg.image.height

            ids = []
            corners_list = []
            radiis = []
            grid_points = {}

            for tag in tagsMsg.tags:
                ids.append(tag.id)
                radiis.append(1.0)
                corners = self.decode_tag_corners(tag, img_width, image_height)
                corners_list.append(corners)
                if tag.gridId not in grid_points:
                    grid_points[tag.gridId] = []
                grid_points[tag.gridId].append(corners)

            for grid_id in grid_points:
                points = np.array(grid_points[grid_id]).flatten().reshape(-1, 2)
                hull = ConvexHull(points)
                # add start point to the end
                hull_points = points[hull.vertices, :]
                hull_points = np.vstack((hull_points, hull_points[0, :]))
                rr.log(topic_name+"/grid", rr.LineStrips2D([hull_points.tolist()], class_ids=[grid_id], radii=[1]))

            rr.log(topic_name, rr.LineStrips2D(corners_list, class_ids=ids, radii=radiis))
            # https://ref.rerun.io/docs/python/0.11.0/common/archetypes/#rerun.archetypes.LineStrips2D

            if not self.queue.full():
                stamp_ns = tagsMsg.header.stamp
                self.queue.put((topic_name, stamp_ns, msg))

    def decode_tag_corners(self, tag, img_width, img_height):
        corners = np.zeros((5,2))
        corners[:4, :] = np.array(tag.pointsPolygon).reshape(4,2)
        # corners[0, :] = [tag.pointsPolygon.pt1.x, tag.pointsPolygon.pt1.y] # [u, v]
        # corners[1, :] = [tag.pointsPolygon.pt2.x, tag.pointsPolygon.pt2.y]
        # corners[2, :] = [tag.pointsPolygon.pt3.x, tag.pointsPolygon.pt3.y]
        # corners[3, :] = [tag.pointsPolygon.pt4.x, tag.pointsPolygon.pt4.y]

        # convert percentage to image size for visualisation
        corners[:, 0] *= img_width
        corners[:, 1] *= img_height
        corners[4, :] = corners[0, :]

        return corners.tolist()
    
    def writer_thread(self):

        self.stopping = False

        # Get the current date and time
        now = datetime.datetime.now()

        # Format the date and time as a string
        date_string = now.strftime("%Y%m%d_%H%M")

        filename = f"mcap_recording_{date_string}.vbag"

        with open(filename, "wb") as f:
            while self.stopping == False:
                try:
                    topic_name, stamp_ns, msg_bytes = self.queue.get(timeout=1)
                except queue.Empty:
                    continue

                # initialisation
                if self.writer is None:
                    self.writer = Writer(f)
                    print(f"initialised writer to log tag detection into file {filename}")

                    self.writer.start()
                    self.schema_id = self.writer.register_schema(name="TagDetections", encoding="capnproto", data="".encode('utf-8'))
                    
                if topic_name not in self.channel_ids:
                    self.channel_ids[topic_name] = self.writer.register_channel(schema_id=self.schema_id, topic=topic_name, message_encoding="capnproto")
                
                self.writer.add_message(channel_id=self.channel_ids[topic_name], log_time=stamp_ns, publish_time=stamp_ns, data=msg_bytes)

            self.writer.finish()
            print("writer finishes")
            

    def stop_writer(self):
        self.stopping = True
        print("stopping set to true")

class OdometryLogeer:
    def __init__(self, topic_list) -> None:
        self.subs = []
        self.counts = {}
        self.tracks = {}
        self.total_distance = {}
        for topic in topic_list:
            sub = CapnpSubscriber("Odometry3d", topic)
            self.subs.append(sub)
            sub.set_callback(self.callback)

    def callback(self, topic_type, topic_name, msg, ts):
        # print(f"callback of topic {topic_name}")

        with eCALOdometry3d.Odometry3d.from_bytes(msg) as odomMsg:
            t_body = [odomMsg.pose.position.x, odomMsg.pose.position.y, odomMsg.pose.position.z]
            q_body = [odomMsg.pose.orientation.x, odomMsg.pose.orientation.y, odomMsg.pose.orientation.z, odomMsg.pose.orientation.w]

            # print(f"position = {odomMsg.pose.position.x}, {odomMsg.pose.position.y}, {odomMsg.pose.position.z}")
            # print(f"orientation = {odomMsg.pose.orientation.w}, {odomMsg.pose.orientation.x}, {odomMsg.pose.orientation.y}, {odomMsg.pose.orientation.z}")

            if topic_name not in self.counts:
                self.counts[topic_name] = 0
                self.tracks[topic_name] = []
                self.tracks[topic_name].append(t_body)
                self.total_distance[topic_name] = 0
                rr.log("S0", rr.ViewCoordinates.FLU, timeless=True)
            else:
                # check for distance, do not draw for <0.05m
                distance = np.linalg.norm(np.array(self.tracks[topic_name][-1]) - np.array(t_body))

                if distance > 0.05:
                    self.tracks[topic_name].append(t_body)
                    self.total_distance[topic_name] += distance
                    # print(self.tracks[topic_name])
                    rr.log("S0/tracks", rr.LineStrips3D(self.tracks[topic_name]))

            rr.set_time_nanos("host_monotonic_time", odomMsg.header.stamp)
            rr.log("S0/body", rr.Transform3D(translation=t_body, rotation=rr.Quaternion(xyzw=q_body)))

            if (self.counts[topic_name] % 50) == 0:
                rr.log("logs", rr.TextLog(f"current pos [{t_body[0]:.2f}, {t_body[1]:.2f}, {t_body[2]:.2f}], total distance = {self.total_distance[topic_name]:.1f}", level=rr.TextLogLevel.DEBUG))

            self.counts[topic_name] += 1

        #     rr.log(
        #     "camera", rr.Transform3D(translation=image.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True)
        # )
        # rr.log("camera", rr.ViewCoordinates.RDF, timeless=True)  # X=Right, Y=Down, Z=Forward


def main():  

    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))
    
    # initialize eCAL API
    ecal_core.initialize(sys.argv, "vk_viewer")
    
    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")

    rr.init("vk_viewer_rr")
    rr.spawn(memory_limit='10%')
    # rr.serve()
    rr.set_time_seconds("host_monotonic_time", time.monotonic_ns())

    rr.log("S0", rr.ViewCoordinates.FLU, timeless=True)
    points = []

    # draw grid with a single connected lines
    min_x = -50
    max_x = 50
    min_y = -50
    max_y = 50
    step = 5

    even_line = True
    for x in range(min_x, max_x + step, step):
        if even_line:
            line = [[x, max_y, 0], [x, min_y, 0]]
        else:
            line = [[x, min_y, 0], [x, max_y, 0]]
        even_line = not even_line
        points += line

    even_line = True
    for y in range(min_y, max_y + step, step):
        if even_line:
            line = [[min_x, y, 0], [max_x, y, 0]]
        else:
            line = [[max_x, y, 0], [min_x, y, 0]]
        even_line = not even_line
        points += line

    rr.log("S0/grid", rr.LineStrips3D(points, radii=[0.01]))

    image_logger = ImageLogger(["S0/camb", "S0/camc", "S0/camd"])
    tags_logger = TagDetectionLogger(["S0/camb/tag_detection", "S0/camc/tag_detection", "S0/camd/tags"])
    odometry_logger = OdometryLogeer(["S0/vio_odom"])


    def handler(signum, frame):
        print("ctrl-c is pressed")
        tags_logger.stop_writer() # to stop gracefully
        tags_logger.thread.join()
        
        print("Exit gracefully")
        exit(0)

    signal.signal(signal.SIGINT, handler)
    
    # idle main thread
    while ecal_core.ok():
        # imuMsg = imuSub.pop_queue()
        # print(f"seq = {imuMsg.header.seq}")
        # print(f"latency device = {imuMsg.header.latencyDevice / 1e6} ms")
        # print(f"latency host = {imuMsg.header.latencyHost / 1e6} ms")
        # accel = np.array([imuMsg.linearAcceleration.x, imuMsg.linearAcceleration.y, imuMsg.linearAcceleration.z])
        # gyro = np.array([imuMsg.angularVelocity.x, imuMsg.angularVelocity.y, imuMsg.angularVelocity.z])
        # print(f"accel = {accel}")
        # print(f"gyro = {gyro}")
        time.sleep(0.1)

        if False:

            msg = eCALTagDetection.TagDetections.new_message()

            msg.header.stamp = time.monotonic_ns()

            msg.image.width = 1280
            msg.image.height = 800

            tags = msg.init('tags', 1)
            tags[0].id = 1
            tags[0].pointsPolygon = [0.05, 0.05, 0.05, 0.09, 0.09, 0.09, 0.09, 0.05]

            tags_logger.callback("TagDetections", "S0/camb/tag_detection", msg.to_bytes(), time.monotonic_ns())
        
    
    # finalize eCAL API
    ecal_core.finalize()

if __name__ == "__main__":
    main()
