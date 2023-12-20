#!/usr/bin/env python3

import sys
import time

import capnp
import numpy as np
import cv2

import ecal.core.core as ecal_core

capnp.add_import_hook(['thirdparty/ecal_common/src/capnp'])

import image_capnp as eCALImage
import imu_capnp as eCALImu

from thirdparty.ecal_common.python.capnp_subscriber import CapnpSubscriber

# visualisation tools
import rerun as rr

class ImageLogger:
    def __init__(self, topic_list, rr) -> None:
        self.rr = rr
        self.subs = []
        for topic in topic_list:
            sub = CapnpSubscriber("Image", topic)
            self.subs.append(sub)
            sub.set_callback(self.callback)

    def callback(self, topic_type, topic_name, msg, ts):
        print(f"callback of topic {topic_name}")

        with eCALImage.Image.from_bytes(msg) as imageMsg:

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
        
        # we have to ways to prevent all images drawn to the same screen
        rr.log(topic_name + "/image", rr.DisconnectedSpace())
        # rr.log(topic_name + "/image", rr.Pinhole(focal_length=300, width=imageMsg.width*2//3, height=imageMsg.height), timeless=True)
        
        rr.log(topic_name + "/image", rr.Image(mat[:, :mat.shape[1]*2//3]))

        # some debugging drawings
        # rr.log(topic_name + "/image", rr.Boxes2D(mins=[10,20], sizes=[20,40]))
        # rr.log(topic_name + "/image", rr.LineStrips2D([[[100, 100], [200,300], [400, 100], [600, 200]]]))
        



def main():  

    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))
    
    # initialize eCAL API
    ecal_core.initialize(sys.argv, "vk_viewer")
    
    # set process state
    ecal_core.set_process_state(1, 1, "I feel good")

    rr.init("vk_viewer_rr")
    rr.spawn(memory_limit='25%')

    image_logger = ImageLogger(["S0/camb", "S0/camc", "S0/camd"], rr)
    
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
        
    
    # finalize eCAL API
    ecal_core.finalize()

if __name__ == "__main__":
    main()
