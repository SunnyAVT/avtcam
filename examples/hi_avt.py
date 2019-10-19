import pymba
import numpy
from ..jetavtcam.avt_camera import AVTCamera
import ipywidgets
from IPython.display import display
from ..jetavtcam.utils import bgr8_to_jpeg


def main():
    camera = AVTCamera(width=224, height=224, capture_width=640, capture_height=480, capture_device=0)
    image = camera.read()

    print(image.shape)
    print(camera.value.shape)


    image_widget = ipywidgets.Image(format='jpeg')
    image_widget.value = bgr8_to_jpeg(image)

    display(image_widget)

    def update_image(change):
        image = change['new']
        image_widget.value = bgr8_to_jpeg(image)

    camera.running = True
    camera.observe(update_image, names='value')
    # camera.unobserve(update_image, names='value')


if __name__ == '__main__':
    main()