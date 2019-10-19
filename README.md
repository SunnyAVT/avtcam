# avtcam

avtcam is an easy to use Python camera interface for NVIDIA Jetson or other embedded platform.

*  Works with various [Allied Vision Camera](www.alliedvision.com) in Jetson Nano or other embedded boards.

*  Easily read images as ``numpy`` arrays with ``image = camera.read()``

*  Set the camera to ``running = True`` to attach callbacks to new frames

avtcam makes it easy to prototype AI projects in Python, especially within the Jupyter Lab programming environment installed in https://github.com/SunnyAVT/avtcam/examples.


## Setup

```bash
git clone https://github.com/SunnyAVT/avtcam
cd avtcam
sudo python3 setup.py install
```

> avtcam is tested against a system configured with the https://github.com/SunnyAVT/avtcam/examples .  Different system configurations may require additional steps.

## Usage

Below we show some usage examples.  You can find more in the [notebooks](notebooks).

### Create Allied Vision camera

Call ``AVTCamera`` to use a compatible Allied Vision camera.  ``capture_width``, ``capture_height``, and ``capture_fps`` will control the capture shape and rate that images are aquired.  ``width`` and ``height`` control the final output shape of the image as returned by the ``read`` function.

```python
from avtcam.avt_camera import AVTCamera

camera = AVTCamera(width=224, height=224, capture_width=1024, capture_height=768, capture_device=0)
```


### Read

Call ``read()`` to read the latest image as a ``numpy.ndarray`` of data type ``np.uint8`` and shape ``(224, 224, 3)``.  The color format is ``BGR8``.

```python
image = camera.read()
```

The ``read`` function also updates the camera's internal ``value`` attribute.

```python
camera.read()
image = camera.value
```

### Callback

You can also set the camera to ``running = True``, which will spawn a thread that acquires images from the camera.  These will update the camera's ``value`` attribute automatically.  You can attach a callback to the value using the [traitlets](https://traitlets.readthedocs.io/en/stable/api.html#callbacks-when-trait-attributes-change) library.  This will call the callback with the new camera value as well as the old camera value

```python
camera.running = True

def callback(change):
    new_image = change['new']
    # do some processing...

camera.observe(callback, names='value')
```

## Cameras

### Allied Vision Cameras

These cameras work with the [``AVTCamera``](avtcam/avt_camera.py) class.  Try them out by following the example [notebook](avtcam/examples/avt_camera.ipynb).

| Model | Infared | 
|:-------|:-----:|
| [EMBEDDED VISION - ALLIED VISION'S NEW ALVIUM CAMERA SERIES](https://www.alliedvision.com/en/products/embedded-vision-cameras.html) | 
| [MACHINE VISION CAMERAS](https://www.alliedvision.com/en/products/cameras.html) | 


## See also

- [jetson_nano_avtcamera_tracking](https://github.com/SunnyAVT/jetson_nano_avtcamera_tracking) - Based on jetson nano detection and tracking project, develop a AVT camera version tracking project.


