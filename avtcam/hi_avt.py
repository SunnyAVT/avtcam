import cv2
import time
from avt_camera import AVTCamera

'''
# Display with image_widget in Jupyter
import ipywidgets
from IPython.display import display
from utils import bgr8_to_jpeg
'''

update_flag = False
image = None
def main():
    global update_flag
    global image
    camera = AVTCamera(width=224, height=224, capture_width=640, capture_height=680, capture_device=0)
    image = camera.read()

    print("image: ", image.shape)
    print("callback return - ", camera.value.shape)

    cv2.imshow("cam", image)
    c = cv2.waitKey(100)
    update_flag = False

    '''
    # Display with image_widget in Jupyter
    image_widget = ipywidgets.Image(format='jpeg')
    image_widget.value = bgr8_to_jpeg(image)
    display(image_widget)
    '''

    def update_image(change):
        global update_flag
        global image
        image = change['new']
        update_flag = True

        '''
        # Display with image_widget in Jupyter
        image_widget.value = bgr8_to_jpeg(image)
        '''

        print("update:", image.shape, image[100, 100, 0])

        # Warning: the display in the callback will lead to trashed frame issue
        #cv2.imshow("cam-live", image)
        #c = cv2.waitKey(5)


    camera.running = True
    camera.observe(update_image, names='value')

    while True:
        if update_flag and camera.running:
            cv2.imshow("cam-live", image)
            #print("show", image[100, 100, 0])
            c = cv2.waitKey(20)
            update_flag = False
        else:
            #time.sleep(0.1)
            c = cv2.waitKey(100)
            if c == ord('q') or c == 27 :
                break
    time.sleep(1)
    camera.unobserve(update_image, names='value')
    cv2.destroyAllWindows()
    print("program exit...")

if __name__ == '__main__':
    main()

