import cv2
import queue
from threading import Lock

# from capnp_subscriber import CapnpSubscriber
from capnp_subscriber import CapnpSubscriber

class SyncedSubscriber:
    def __init__(self, types, topics, 
                 typeclasses=None, 
                 enforce_sync=True):

        self.subscribers = {}
        # store individual incoming messages
        self.queues = {}
        # store properly synced messages
        self.synced_queue = queue.Queue(5)
        self.enforce_sync = enforce_sync
        self.callbacks = []
        
        self.latest = None
        assert len(types) == len(topics)
        assert typeclasses is None or len(typeclasses) == len(topics)

        if enforce_sync:
            assert typeclasses is not None
            
        self.size = len(topics)

        self.rolling = False

        self.assemble = {}
        self.assemble_index = -1

        self.lock = Lock()

        for i in range(len(topics)):
            print(f"subscribing to {types[i]} topic {topics[i]}")
            typeclass = typeclasses[i] if typeclasses is not None else None
            sub = self.subscribers[topics[i]] = CapnpSubscriber(types[i], topics[i], typeclass)
            sub.set_callback(self.callback)

            self.queues[topics[i]] = queue.Queue(10)

    def queue_update(self):
        for queueName in self.queues:
            m_queue = self.queues[queueName]

            # already in assemble, no need to get from queue
            if queueName in self.assemble:
                continue

            while True:                  

                if m_queue.empty():
                    break

                genericMsg, msgRaw = m_queue.get()

                if self.enforce_sync and self.assemble_index < genericMsg.header.stamp:
                    # we shall throw away the assemble and start again
                    # if self.assemble_index != -1:
                        # print(f"reset index to {genericMsg.header.stamp}")

                    self.assemble_index = genericMsg.header.stamp
                    self.assemble = {}
                    self.assemble[queueName] = (genericMsg, msgRaw)
                    
                    continue
                elif self.enforce_sync and self.assemble_index > genericMsg.header.stamp:
                    # print(f"ignore {queueName} for later")
                    continue
                else:
                    self.assemble[queueName] = (genericMsg, msgRaw)
                    if self.enforce_sync:
                        break        
        
        # check for full assembly
        if len(self.assemble) == self.size:
            self.latest = self.assemble

            for cb in self.callbacks:
                cb(self.latest, self.assemble_index)

            if self.rolling:
                if self.synced_queue.full():
                    print(f"queue full: {self.queues.keys()}")
                    self.synced_queue.get()
                    self.synced_queue.put(self.assemble)
                else:
                    self.synced_queue.put(self.assemble, block=False)
            self.assemble = {}
            self.assemble_index = -1

    def callback(self, topic_type, topic_name, msg, ts, msg_raw):
        if self.queues[topic_name].full():
            self.queues[topic_name].get()
        
        self.queues[topic_name].put((msg, msg_raw))

        with self.lock:
            self.queue_update()

    # sub must have a `register_callback()` method
    def add_external_sub(self, sub, topic_name):
        self.queues[topic_name] = queue.Queue(10)
        self.size += 1
        sub.register_callback(self.callback)

    def register_callback(self, cb):
        self.callbacks.append(cb)

    def pop_latest(self):
        with self.lock:
            if self.latest == None:
                return {}
            else:
                return self.latest

    def pop_sync_queue(self):
        # not protected for read
        return self.synced_queue.get()
    
    def pop_sync_queue_realtime(self):
        
        ret = self.synced_queue.get()

        while not self.synced_queue.empty():
            ret = self.synced_queue.get()
        return ret 