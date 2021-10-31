# Blaze Face Detection refer to https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference
import cv2
import numpy as np
from .blazeFaceDetector import blazeFaceDetector


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def blaze_detect(image_rgb, scoreThreshold = 0.7, iouThreshold = 0.3, modelType = "back"):
    # Initialize face detector
    h, w, c = image_rgb.shape
    faceDetector = blazeFaceDetector(modelType, scoreThreshold, iouThreshold)
    # Detect faces
    results = faceDetector.detectFaces(image_rgb)

    # # Draw detections
    # img_plot = faceDetector.drawDetections(img, detectionResults)
    # cv2.imshow("detections", img_plot)
    # cv2.waitKey(0)
    boundingBoxes = results.boxes
    keypoints = results.keypoints
    scores = results.scores
    bboxes = []

    # Add bounding boxes and keypoints
    for boundingBox, keypoints, score in zip(boundingBoxes, keypoints, scores):
        x1 = (w * boundingBox[0]).astype(int)
        x2 = (w * boundingBox[2]).astype(int)
        y1 = (h * boundingBox[1]).astype(int)
        y2 = (h * boundingBox[3]).astype(int)
        bboxes.append([x1, y1, x2, y2])
    if len(bboxes) > 0:
        bboxes = np.array(bboxes).astype('int')
        bboxes = xyxy_to_xywh(np.array(bboxes))
        bboxes[:, 2] = bboxes[:, 2] * 1.1
        bboxes[:, 3] = bboxes[:, 3] * 1.1
    else:
        bboxes = None
    return bboxes