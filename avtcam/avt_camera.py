import atexit
import cv2
import numpy as np
import traitlets
from pymba import Vimba, VimbaException, Frame


class AVTCamera(traitlets.HasTraits):
    # global vimba and vimba cameras group
    vimba = None
    cameras = None

    value = traitlets.Any()
    width = traitlets.Integer(default_value=224)
    height = traitlets.Integer(default_value=224)
    format = traitlets.Unicode(default_value='bgr8')   # the format is fixed now
    running = traitlets.Bool(default_value=False)

    capture_fps = traitlets.Integer(default_value=10)   # wait to support it later
    capture_width = traitlets.Integer(default_value=640)
    capture_height = traitlets.Integer(default_value=480)   
    capture_device = traitlets.Integer(default_value=0)

    def __init__(self, *args, **kwargs):
        super(AVTCamera, self).__init__(*args, **kwargs)
        self.curCam = None
        self._running = False
        self.AcqMode = 'Unknown'
        if self.format == 'bgr8':
            self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)

        self.init_vimba()
        self.init_cam()
        self.getFrame()

        atexit.register(self.release_vimba)
        atexit.register(self.release_cam)


    @traitlets.observe('running')
    def _on_streaming(self, change):
        # change is running status here
        if change['new'] and not change['old']:
            # transition from not running -> running
            self._running = True
            try:
                self.curCam.disarm()
                self.curCam.arm('Continuous', self.frame_callback)
                self.AcqMode = 'Continuous'
                self.curCam.start_frame_acquisition()
            except VimbaException as e:
                print(e)
                raise RuntimeError('Camera start Continuous running failure !!!')
        elif change['old'] and not change['new']:
            # transition from running -> not running
            self._running = False
            #self.thread.join()
            self.curCam.disarm()
            self.AcqMode = 'Unknown'


    def frame_callback(self, frame: Frame) -> None:
        type, image_8bit = self.convertFrame(frame)
        image = self.fillUserFrame(type, image_8bit)
        self.value = cv2.resize(image, (int(self.width), int(self.height)))
        print("avt frame_callback")


    def fillUserFrame(self, type, frame):
        PIXEL_FORMATS_RGB_CONVERSIONS = {
            'BayerRG8': cv2.COLOR_BAYER_RG2RGB,
        }
        PIXEL_FORMATS_BGR_CONVERSIONS = {
            'BayerRG8': cv2.COLOR_BAYER_RG2BGR,
            'BayerGR8': cv2.COLOR_BAYER_GR2BGR,
            'BayerRG12': cv2.COLOR_BAYER_RG2BGR,
            'BayerRG12Packed': cv2.COLOR_BAYER_RG2BGR,
            'BayerGR12Packed': cv2.COLOR_BAYER_GR2BGR,
            'Mono8': cv2.COLOR_GRAY2BGR,
            'Mono10': cv2.COLOR_GRAY2BGR,
            'Mono12': cv2.COLOR_GRAY2BGR,
            'Mono14': cv2.COLOR_GRAY2BGR,
            'RGB8Packed': cv2.COLOR_RGB2BGR
        }

        if self.format == 'bgr8':
            if (type == "BGR8Packed"):
                img = frame
            else:
                img = cv2.cvtColor(frame, PIXEL_FORMATS_BGR_CONVERSIONS[type])
            return img

    def convertFrame(self, frame):
        camera_frame_size = len(frame.buffer_data())
        frame_pixel_format = frame.pixel_format
        Width = self.capture_width
        Height = self.capture_height
        print("Resolution: %dx%d, FrameSize: %d, PixelFormat: %s" %(Width, Height, camera_frame_size, frame_pixel_format))
        data_bytes = frame.buffer_data()
        if (frame_pixel_format == "Mono8" or frame_pixel_format == "BayerRG8" or frame_pixel_format == "BayerGR8"):
            frame_8bits = np.ndarray(buffer=data_bytes, dtype=np.uint8, shape=(Height, Width))
        elif (frame_pixel_format == "BayerRG12" or frame_pixel_format == "Mono10" or frame_pixel_format == "Mono12" or frame_pixel_format == "Mono14"):
            data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
            pixel_even = data_bytes[0::2]
            pixel_odd = data_bytes[1::2]
            # Convert bayer16 to bayer8 / Convert Mono12/Mono14 to Mono8
            if (frame_pixel_format == "Mono14"):
                pixel_even = np.right_shift(pixel_even, 6)
                pixel_odd = np.left_shift(pixel_odd, 2)
            elif (frame_pixel_format == "Mono10"):
                pixel_even = np.right_shift(pixel_even, 2)
                pixel_odd = np.left_shift(pixel_odd, 6)
            else:
                pixel_even = np.right_shift(pixel_even, 4)
                pixel_odd = np.left_shift(pixel_odd, 4)
            frame_8bits = np.bitwise_or(pixel_even, pixel_odd).reshape(Height, Width)
        elif (frame_pixel_format == "BayerRG12Packed" or frame_pixel_format == "Mono12Packed" or frame_pixel_format == "BayerGR12Packed"):
            data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
            size = len(data_bytes)
            index = []
            for i in range(0, size, 3):
                index.append(i + 1)
            data_bytes = np.delete(data_bytes, index)
            frame_8bits = data_bytes.reshape(Height, Width)
        elif (frame_pixel_format == "RGB8Packed" or frame_pixel_format == "BGR8Packed"):
            frame_8bits = np.ndarray(buffer=frame.buffer_data(), dtype=np.uint8, shape=(Height, Width * 3))
        else:
            # Note: wait to do -- other format, such as YUV411Packed, YUV422Packed, YUV444Packed
            frame_8bits = np.ndarray(buffer=frame.buffer_data(), dtype=np.uint8, shape=(Height, Width))
            raise RuntimeError('Unsupported image format, please re-configurate the camera!!!')
        return frame_pixel_format, frame_8bits

    def _read(self):
        type, image_8bit = self.getFrame()
        image = self.fillUserFrame(type, image_8bit)
        image_resized = cv2.resize(image, (int(self.width), int(self.height)))
        return image_resized

    def read(self):
        if self._running:
            raise RuntimeError('Cannot read directly while camera is running')
        self.value = self._read()
        return self.value

    def getFrame(self):
        if self._running:
            raise RuntimeError('Cannot read directly while camera is running')
        else:
            try:
                if self.AcqMode is not 'SingleFrame':
                    self.curCam.arm('SingleFrame')
                    self.AcqMode = 'SingleFrame'
                raw_frame = self.curCam.acquire_frame()
                type, frame = self.convertFrame(raw_frame)
            except VimbaException as e:
                print(e)
                raise RuntimeError("acquire_frame error")
            return type, frame

    def _setROI(self):
        try:
            feature_h = self.curCam.feature("Height")
            feature_h.value = self.capture_height
            feature_w = self.curCam.feature("Width")
            feature_w.value = self.capture_width
        except:
            raise RuntimeError("Wrong capture_height/capture_width !!'acquire_frame'!")

    def init_cam(self):
        vmFactory = AVTCamera.vimba.camera_ids()
        # Get connected cameras
        AVTCamera.cameras = [AVTCamera.vimba.camera(id) for id in vmFactory]
        cam_nums = len(AVTCamera.cameras);
        if cam_nums == 0:
            raise RuntimeError("Warning: No camera present.")
        elif self.capture_device >= cam_nums:
            raise RuntimeError("Warning: No specified camera.")
        else:
            self.curCam = AVTCamera.cameras[self.capture_device]
            try:
                self.curCam.open()
                self._setROI()
                self.curCam.arm('SingleFrame')
                self.AcqMode = 'SingleFrame'
                print("Device {} open OK".format(self.capture_device))
            except VimbaException as e:
                print(e)
                raise RuntimeError("init_cam camera error")


    def release_cam(self):
        if self.curCam:
            print("Release: Device {} ID: {}".format(self.capture_device, self.curCam))
            self.curCam.disarm()
            self.curCam.close()

    def init_vimba(self):
        if AVTCamera.vimba == None:
            AVTCamera.vimba = Vimba()
            AVTCamera.vimba.startup()
            print ("init_vimba")

    def release_vimba(self):
        if AVTCamera.vimba:
            AVTCamera.vimba.shutdown()
            AVTCamera.vimba = None
            print ("Exit: release_vimba")