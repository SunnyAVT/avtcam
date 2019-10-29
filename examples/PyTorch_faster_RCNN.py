# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import time
from avtcam import AVTCamera
from avtcam import bgr8_to_jpeg


def get_prediction(img_path, threshold):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    """
    object_detection_api
        parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
        method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written
          with opencv
        - the final image is displayed
    """
    boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


'''
# download an image for inference
!wget https://www.wsha.org/wp-content/uploads/banner-diverse-group-of-people-2.jpg -O people.jpg

# use the api pipeline for object detection
# the threshold is set manually, the model sometimes predict
# random structures as some object, so we set a threshold to filter
# better prediction scores.
object_detection_api('./people.jpg', threshold=0.8)

!wget https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/10best-cars-group-cropped-1542126037.jpg -O cars.jpg
object_detection_api('./cars.jpg', rect_th=6, text_th=5, text_size=5)

!wget https://images.unsplash.com/photo-1458169495136-854e4c39548a -O traffic_scene2.jpg
object_detection_api('./traffic_scene2.jpg', rect_th=15, text_th=7, text_size=5, threshold=0.8)

!wget https://images.unsplash.com/photo-1458169495136-854e4c39548a -O girl_cars.jpg
object_detection_api('./girl_cars.jpg', rect_th=15, text_th=7, text_size=5, threshold=0.8)
'''

def check_inference_time(image_path, gpu=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    img = Image.open(image_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    if gpu:
        model.cuda()
        img = img.cuda()
    else:
        model.cpu()
        img = img.cpu()
    start_time = time.time()
    pred = model([img])
    end_time = time.time()
    return end_time-start_time



update_flag = False
image = None
# get the pretrained model from torchvision.models
# Note: pretrained=True will get the pretrained weights for the model.
# model.eval() to use the model for inference
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Class labels from official PyTorch documentation for the pretrained model
# Note that there are some N/A's
# for complete list check https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# we will use the same list for this notebook
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def update_image(change):
    global update_flag
    global image
    image = change['new']
    update_flag = True
    # print("update:", image.shape, image[100, 100, 0])

    '''
    # Display with image_widget in Jupyter
    image_widget.value = bgr8_to_jpeg(image)
    '''
    # Warning: the display in the callback will lead to trashed frame issue
    # cv2.imshow("cam-live", image)
    # c = cv2.waitKey(5)

def main():
    global update_flag
    global image
    global model
    # capture_width/capture_height is the ROI of the camera
    # width/height is the size of image returned by avtcam module
    camera = AVTCamera(width=800, height=600, capture_width=1024, capture_height=768, capture_device=0)
    update_flag = False

    '''
    image = camera.read()
    print("image: ", image.shape)
    print("callback return - ", camera.value.shape)
    cv2.imshow("SingleFrame", image)
    c = cv2.waitKey(100)
    '''

    camera.running = True
    camera.observe(update_image, names='value')

    while True:
        if update_flag and camera.running:
            #cv2.imshow("cam-live", image)
            # print("show", image[100, 100, 0])
            object_detection_live(model, image, rect_th=2, text_th=1, text_size=1, threshold=0.4)
            c = cv2.waitKey(10)
            update_flag = False

            if c == ord('q') or c == 27:
                c = cv2.waitKey(200)
                break
            # capture one image with keyin 'c'
            elif c == ord('c'):
                cv2.imshow("Snapshot", image)
                cv2.waitKey(10)
                '''
                cap_image = bgr8_to_jpeg(image)
                f = open('/home/capture.jpg', 'wb')
                f.write(bytearray(cap_image))
                f.close()
                '''

        else:
            c = cv2.waitKey(50)
            if c == ord('q') or c == 27:
                break

    time.sleep(1)
    camera.unobserve(update_image, names='value')
    cv2.destroyAllWindows()
    print("program exit...")

def object_detection_live(model, img, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())

    #print("debug: pred_class=", pred_class)
    #print("debug: pred_boxes=", pred_boxes)
    #print("debug: pred_score=", pred_score)

    display_box = True
    #pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_t_matched = [pred_score.index(x) for x in pred_score if x > threshold]
    if(len(pred_t_matched)>0):
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    else:
        pred_t = 0
        display_box = False
    #print("debug: matched prediction number = ", pred_t)

    if display_box:
        boxes = pred_boxes[:pred_t]
        pred_cls = pred_class[:pred_t]

    img = np.transpose(img.numpy(), (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if display_box:
        for i in range(len(boxes)):
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    cv2.imshow("live-detection", img)
    #c = cv2.waitKey(20)

if __name__ == '__main__':
    main()

    #object_detection_api('./girl_cars.jpg', rect_th=15, text_th=7, text_size=5, threshold=0.8)
    #cpu_time = sum([check_inference_time('./girl_cars.jpg', gpu=False) for _ in range(10)])/10.0
    #gpu_time = sum([check_inference_time('./girl_cars.jpg', gpu=True) for _ in range(10)])/10.0

    #print('\n\nAverage Time take by the model with GPU = {}s\nAverage Time take by the model with CPU = {}s'.format(gpu_time, cpu_time))