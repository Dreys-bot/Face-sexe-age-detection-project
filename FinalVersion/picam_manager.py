#!/usr/bin/env python3

from picamera2 import Picamera2, Preview

class PiCam():
    def __init__(self, normal_size = (640, 480), lowres_size = (320, 240), callback = None, preview = True):
        self.normal = normal_size
        self.lowres = lowres_size

        self.cam = Picamera2()

        if preview:
            self.cam.start_preview(Preview.QTGL)
            config = self.cam.create_preview_configuration(main={"size": normal_size},lores={"size": lowres_size, "format": "YUV420"})
            self.cam.configure(config)

        self.stride = self.cam.stream_configuration("lores")["stride"]
        if callback is not None:
            self.cam.post_callback = callback
        self.cam.start()
        
    def get_buffer(self):
        return self.cam.capture_buffer("lores")
        
    def get_gray(self):
        buff = self.cam.capture_buffer("lores")
        return buff[:self.stride * self.lowres[1]].reshape((self.lowres[1], self.stride))
        
    def stop(self):
        self.cam.stop_preview()
        self.cam.stop()
