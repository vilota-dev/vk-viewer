#!/usr/bin/env python3

import sys
import time

import capnp
import numpy as np
import cv2
from scipy.spatial import ConvexHull

import ecal.core.core as ecal_core

capnp.add_import_hook(['thirdparty/vk_common/capnp'])

import image_capnp as eCALImage
import imu_capnp as eCALImu
import tagdetection_capnp as eCALTagDetection
import odometry3d_capnp as eCALOdometry3d
import flow2d_capnp as eCALFlow2d

from capnp_subscriber import CapnpSubscriber

from sync_utils import SyncedSubscriber

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
            
            if imageMsg.mipMapLevels == 0:
                rr.log(topic_name, rr.Image(mat[:, :mat.shape[1]]))
            else:
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
    
def remove_after_last_slash(input_string):
    last_slash_index = input_string.rfind("/")
    if last_slash_index != -1:
        result_string = input_string[:last_slash_index]
        return result_string
    else:
        # If there is no "/" in the string, return the original string
        return input_string

    
class TagDetectionLogger:
    def __init__(self, topic_list) -> None:
        self.subs = []
        self.tag_frame_count = 0
        self.grid_frame_count = 0
        for topic in topic_list:
            types = ["TagDetections", "Image"]
            topics = [topic, remove_after_last_slash(topic)]
            type_classes = [eCALTagDetection.TagDetections, eCALImage.Image]

            sub = SyncedSubscriber(types, topics, type_classes)
            self.subs.append(sub)

            sub.register_callback(self.callback)

            # sub = CapnpSubscriber("TagDetections", topic)
            # self.subs.append(sub)
            # sub.set_callback(self.callback)

        # just to make some decoupling of the disk writing and visualisation
        self.queue = queue.Queue(10)
        self.writer = None
        self.channel_ids = dict()

        self.thread = threading.Thread(target=self.writer_thread)
        self.thread.start()

    def callback(self, assemble, index):

        has_tags = False
        has_grids = False

        # skip non detection for recording and viewing
        for topic_name in assemble:
            if "tags" in topic_name:
                tagsMsg, msgRaw = assemble[topic_name]
                for tag in tagsMsg.tags:
                    has_tags = True
                    if tag.gridId > 0: # proper hash checked
                        has_grids = True

        if not has_tags:
            return

        rr.set_time_nanos("host_monotonic_time", index)

        for topic_name in assemble:
            if "tags" in topic_name:

                rr.log(topic_name, rr.DisconnectedSpace())

                tagsMsg, msgRaw = assemble[topic_name]
                # obtain image size
                if tagsMsg.image.mipMapLevels == 0:
                    img_width = tagsMsg.image.width
                else:
                    img_width = tagsMsg.image.width * 2 // 3
                image_height = tagsMsg.image.height

                base_radii = int(min(img_width, image_height) * 0.005)

                ids = []
                corners_list = []
                radiis = []
                grid_points = {}

                for tag in tagsMsg.tags:
                    ids.append(tag.id)
                    radiis.append(base_radii)
                    corners = self.decode_tag_corners(tag, img_width, image_height)
                    corners_list.append(corners)

                    # grid logic
                    if tag.gridId > 0: # 0 means tag has no associated grid
                        if tag.gridId not in grid_points:
                            grid_points[tag.gridId] = []
                        grid_points[tag.gridId].append(corners)

                grid_corners = []
                grid_radiis = []
                grid_classes = []

                # visualising grid
                for grid_id in grid_points:
                    points = np.array(grid_points[grid_id]).flatten().reshape(-1, 2)
                    hull = ConvexHull(points)
                    # add start point to the end
                    hull_points = points[hull.vertices, :]
                    hull_points = np.vstack((hull_points, hull_points[0, :]))
                    grid_corners.append(hull_points.tolist())
                    grid_radiis.append(1.0)
                    grid_classes.append(grid_id)

                rr.log(topic_name+"/grid", rr.LineStrips2D(grid_corners, class_ids=grid_classes, radii=grid_radiis))
                rr.log(topic_name, rr.LineStrips2D(corners_list, class_ids=ids, radii=radiis))
                # https://ref.rerun.io/docs/python/0.11.0/common/archetypes/#rerun.archetypes.LineStrips2D

                if has_tags:
                    self.tag_frame_count += 1
                rr.log(topic_name, rr.AnyValues(tag_frame_count=self.tag_frame_count))

                if has_grids:
                    self.grid_frame_count += 1
                rr.log(topic_name, rr.AnyValues(grid_frame_count=self.grid_frame_count))

                if not self.queue.full():
                    stamp_ns = tagsMsg.header.stamp
                    self.queue.put((topic_name, stamp_ns, msgRaw))
            else: # assume image
                imgMsg, _ = assemble[topic_name]

                mat = ImageLogger.image_msg_to_cvmat(imgMsg)
                
                if imgMsg.mipMapLevels == 0:
                    rr.log(topic_name + "/tags/image", rr.Image(mat))
                else:
                    rr.log(topic_name + "/tags/image", rr.Image(mat[:, :mat.shape[1]*2//3]))

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

            if self.writer is not None:
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


class Flow2dLogger:
    def __init__(self, topic_list) -> None:
        self.subs = []
        for topic in topic_list:
            sub = CapnpSubscriber("HFOpticalFlowResult", topic)
            self.subs.append(sub)
            sub.set_callback(self.callback)

    def callback(self, topic_type, topic_name, msg, ts):
        # print(f"callback of topic {topic_name}")

        with eCALFlow2d.HFOpticalFlowResult.from_bytes(msg) as resultMsg:

            # print(f"HFOpticalFlowResult stamp {resultMsg.header.stamp}")

            rr.set_time_nanos("host_monotonic_time", resultMsg.header.stamp)

            width = resultMsg.image.width
            height = resultMsg.image.height
            if resultMsg.image.mipMapLevels > 0:
                width = width * 2 / 3

            image_topic_name = remove_after_last_slash(topic_name)

            points_with_level = dict()
            ids_with_level = dict()
            colors_with_level = dict()

            predictions_by_id = dict()
            history_pos_by_id = dict()
            history_seq_by_id = dict()

            base_radii = int(min(width, height) * 0.008)

            for flow2d in resultMsg.flowData:

                if flow2d.age < 3:
                    continue

                id = flow2d.id

                if flow2d.level not in points_with_level:
                    points_with_level[flow2d.level] = dict()
                    points_with_level[flow2d.level]["stereo"] = list()
                    points_with_level[flow2d.level]["mono"] = list()
                    ids_with_level[flow2d.level] = dict()
                    ids_with_level[flow2d.level]["stereo"] = list()
                    ids_with_level[flow2d.level]["mono"] = list()
                    color_shift = int(min(255, flow2d.level * 125))
                    colors_with_level[flow2d.level] = { 
                        "mono" : [int(color_shift), int(max(0, 255 - 2 * color_shift)) , int(color_shift / 2), 255],
                        "stereo" : [int(max(0, 255 - 2 * color_shift)), int(color_shift / 2) , int(color_shift) , 255]
                    }


                x = flow2d.position.x * width
                y = flow2d.position.y * height

                # check if it is a stereo point
                is_stereo = (flow2d.detectorMethod == eCALFlow2d.Flow2d.DetectorMethod.sparseStereo)

                if is_stereo:
                    points_with_level[flow2d.level]["stereo"].append([x,y])
                    ids_with_level[flow2d.level]["stereo"].append(flow2d.id % 10000) # control range
                else:
                    points_with_level[flow2d.level]["mono"].append([x,y])
                    ids_with_level[flow2d.level]["mono"].append(flow2d.id % 10000) # control range

                for history_item in flow2d.history:
                    if history_item.seq == 0:
                        predictions_by_id[id] = [history_item.position.x * width, history_item.position.y * height]
                    else:
                        if id not in history_pos_by_id:
                            history_pos_by_id[id] = list()
                            history_pos_by_id[id].append([x, y])
                            history_seq_by_id[id] = list()
                            history_seq_by_id[id].append(0)
                        history_pos_by_id[id].append([history_item.position.x * width, history_item.position.y * height])
                        history_seq_by_id[id].append(history_item.seq)
                    

            for level in points_with_level:

                for type, radii in [("stereo", int(base_radii * 1.5)), ("mono", base_radii)]:
                    points = points_with_level[level][type]
                    # print(f"level {level}, {points}")
                    assert len(ids_with_level[level][type]) == len(points_with_level[level][type])
                    rr.log(topic_name + '/current' + "/" + type, rr.Points2D(points, radii = radii, colors=colors_with_level[level][type], class_ids=ids_with_level[level][type]))
        
            # logging prediction
            predictions = list()
            ids = list()
            for id in predictions_by_id:
                predictions.append(predictions_by_id[id])
                ids.append(id)
            if len(predictions_by_id):
                rr.log(topic_name + '/prediction', rr.Points2D(predictions, radii = base_radii / 2, colors=[255,255,255,255]))
            
            # logging history
            positions = list()
            for id in history_pos_by_id:
                positions.append(history_pos_by_id[id])
            
            rr.log(topic_name + '/history', rr.LineStrips2D(positions, radii=base_radii/4, colors=[255,255,255,255]))

def main():  

    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))
    
    # initialize eCAL API
    ecal_core.initialize(sys.argv, "vk_viewer")
    
    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")

    rr.init("vk_viewer_rr")
    rr.spawn(memory_limit='200MB')
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

    image_logger = ImageLogger(["S0/cama", "S0/camb", "S0/camc", "S0/camd", "S0/stereo1_l" , "S0/stereo2_r"])
    tags_logger = TagDetectionLogger(["S0/cama/tags", "S0/camb/tags", "S0/camc/tags", "S0/camd/tags"])
    odometry_logger = OdometryLogeer(["S0/vio_odom"])
    flow2d_logger = Flow2dLogger(["S0/camd/hfflow","S0/stereo1_l/hfflow" , "S0/stereo2_r/hfflow"])


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
