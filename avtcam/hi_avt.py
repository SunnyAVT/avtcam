import pymba
import numpy
import cv2
import time
import threading
from avt_camera import AVTCamera
import matplotlib.pyplot as plt

import ipywidgets
from IPython.display import display
from utils import bgr8_to_jpeg

def main():
    camera = AVTCamera(width=224, height=224, capture_width=1024, capture_height=768, capture_device=0)
    image = camera.read()

    print("image: ", image.shape)
    print("callback return - ", camera.value.shape)

    cv2.imshow("cam", image)
    c = cv2.waitKey(200)

    plt.imshow(image)
    plt.show()

    '''
    # Display with image_widget in Jupyter
    image_widget = ipywidgets.Image(format='jpeg')
    image_widget.value = bgr8_to_jpeg(image)
    display(image_widget)
    '''

    def capture_frames() -> None:
        while True:
            if not camera.running:
                break
            print("thread ...")
            cv2.imshow("cam-live", image)
            c = cv2.waitKey(20)

    def update_image(change):
        image = change['new']
        # Display with image_widget in Jupyter
        #image_widget.value = bgr8_to_jpeg(image)

        # the display in the callback will lead to trashed frame issue
        print("app", image.shape)
        #plt.imshow(image)
        #plt.show()
        #cv2.imshow("cam-live", image)
        #c = cv2.waitKey(5)



    camera.running = True
    camera.observe(update_image, names='value')

    #app_thread = threading.Thread(target=capture_frames)
    #app_thread.start()
    # camera.unobserve(update_image, names='value')

    while True:
        line = input()
        if line == '\n' or line == '':
            break
        time.sleep(1)

    cv2.destroyAllWindows()
    print("program exit...")

if __name__ == '__main__':
    main()

