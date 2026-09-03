import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logic as logic
from helper import *
import computerVision

# Load COCO class names used by the pretrained YOLO model for object classification
labels_Path = os.path.sep.join([os.path.dirname(os.path.abspath(__file__)), "yolo-coco", "coco.names"])
LABELS = open(labels_Path).read().strip().split("\n")

# Construct absolute paths for YOLO configuration and pretrained weights
weights_Path = os.path.sep.join([os.path.dirname(os.path.abspath(__file__)), "yolo-coco", "yolov3.weights"])
config_Path = os.path.sep.join([os.path.dirname(os.path.abspath(__file__)), "yolo-coco", "yolov3.cfg"])

yolo_net = cv2.dnn.readNetFromDarknet(config_Path, weights_Path)


# Read the input image from disk for object detection processing
def detect(image_path, confidence=0.5, threshold=0.3):
    image = cv2.imread(image_path)
    computerVision.filename = image_path
    result = computerVision.runCV()
    if (result):
        return 10e8
    # Extract only the output layers required for YOLO forward inference
    layer_names = yolo_net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

    # Convert image to a blob and perform forward pass through YOLO network
    # to obtain bounding boxes and confidence scores

    input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    yolo_net.setInput(input_blob)
    start_time = time.time()
    layer_outputs = yolo_net.forward(layer_names)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"[INFO] YOLO inference time: {inference_time:.4f} seconds")

    # Measure YOLO inference time for performance analysis
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # Initialize containers to store detected bounding boxes, scores, and class IDs

    return show_result(layer_outputs, confidence, threshold, image)


def getFrameHelper(videoPath):
    vs = cv2.VideoCapture(videoPath)
    arr = videoPath.split('\\')
    arr = arr[len(arr) - 1]
    arr = arr.split('.')
    hola = arr[0]
    count = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed or count > 700:
            break
        cv2.imwrite(f'frames/{hola}/{count}.jpg', frame)
        count = count + 1


def detectfinal(iter):
    imglist = []
    for i in range(2):
        imglist.append(
            os.path.sep.join([os.path.dirname(os.path.abspath(__file__)), "frames", f'{i + 1}', f'{iter}' + '.jpg']))
        imglist.append(
            os.path.sep.join(
                [os.path.dirname(os.path.abspath(__file__)), "frames", f'{i + 3}', f'{iter + 350}' + '.jpg']))
    finalList = detectFour(imglist)

    return logic.conclusion(finalList)

    # return finalList


def detectFour(imglist):
    ra = []
    for i in range(len(imglist)):
        ra.append(detect(imglist[i]))
    return ra


def show_result(layerOutputs, confidence, threshold, image):
    bounding_boxes = []
    confidence_scores = []
    class_ids = []
    (height, width) = image.shape[:2]

    # Iterate through YOLO output layers to process detections
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence_score = scores[class_id]

            # Ignore low-confidence detections below the defined threshold
            if confidence_score > confidence:
                # Scale normalized bounding box coordinates to image dimensions
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                bounding_boxes.append([x, y, int(width), int(height)])
                confidence_scores.append(float(confidence_score))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping detections
    # to keep only the most confident detections
    selected_indices = cv2.dnn.NMSBoxes(bounding_boxes, confidence_scores, confidence,
                            threshold)

    # ensure at least one detection exists
    objects = {'car', 'truck', 'bus', 'bicycle', 'motorbike'}
    annotations = []
    if len(selected_indices) > 0:
        # loop over the indexes we are keeping
        for i in selected_indices.flatten():
            if LABELS[class_ids[i]] not in objects:
                continue

            # draw a bounding box rectangle and label on the image
            annotations.append(hw_bb(bounding_boxes[i]))

    # Draw detected vehicle bounding boxes and save result image
    draw_im(image, annotations)
    plt.savefig('result.png')
    return len(annotations)
if __name__ == "__main__":
    print(detectfinal(1))
