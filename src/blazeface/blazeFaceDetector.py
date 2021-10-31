import time
import os

import cv2
import numpy as np
import tensorflow as tf
from .blazeFaceUtils import gen_anchors, SsdAnchorsCalculatorOptions

KEY_POINT_SIZE = 6
MAX_FACE_NUM = 100
current_path = os.path.dirname(__file__)

class blazeFaceDetector():

	def __init__(self, type = "front", scoreThreshold = 0.7, iouThreshold = 0.3):
		self.type = type
		self.scoreThreshold = scoreThreshold
		self.iouThreshold = iouThreshold
		self.sigmoidScoreThreshold = np.log(self.scoreThreshold/(1-self.scoreThreshold))
		self.fps = 0
		self.timeLastPrediction = time.time()
		self.frameCounter = 0

		# Initialize model based on model type
		self.initializeModel(type)

		# Generate anchors for model
		self.generateAnchors(type)

	def initializeModel(self, type):
		if type == "front":
			self.interpreter = tf.lite.Interpreter(model_path=os.path.join(current_path, "./weights/face_detection_front.tflite"))
		elif type =="back":
			self.interpreter = tf.lite.Interpreter(model_path=os.path.join(current_path, "./weights/face_detection_back.tflite"))
		self.interpreter.allocate_tensors()

		# Get model info
		self.getModelInputDetails()
		self.getModelOutputDetails()

	def detectFaces(self, image):

		# Prepare image for inference
		input_tensor = self.prepareInputForInference(image)

		# Perform inference on the image
		output0, output1 = self.inference(input_tensor)

		# Filter scores based on the detection scores
		scores, goodDetectionsIndices = self.filterDetections(output1)

		# Extract information of filtered detections
		boxes, keypoints = self.extractDetections(output0, goodDetectionsIndices)

		# Filter results with non-maximum suppression
		detectionResults = self.filterWithNonMaxSupression(boxes, keypoints, scores)

		# Update fps calculator
		self.updateFps()

		return detectionResults

	def updateFps(self):
		updateRate = 1
		self.frameCounter += 1

		# Every updateRate frames calculate the fps based on the ellapsed time
		if self.frameCounter == updateRate:
			timeNow = time.time()
			ellapsedTime = timeNow - self.timeLastPrediction

			self.fps = int(updateRate/ellapsedTime)
			self.frameCounter = 0
			self.timeLastPrediction = timeNow


	def drawDetections(self, img, results):

		boundingBoxes = results.boxes
		keypoints = results.keypoints
		scores = results.scores

		# Add bounding boxes and keypoints
		for boundingBox, keypoints, score in zip(boundingBoxes, keypoints, scores):
			x1 = (self.img_width * boundingBox[0]).astype(int)
			x2 = (self.img_width * boundingBox[2]).astype(int)
			y1 = (self.img_height * boundingBox[1]).astype(int)
			y2 = (self.img_height * boundingBox[3]).astype(int)
			cv2.rectangle(img, (x1, y1), (x2, y2), (22, 22, 250), 2)
			cv2.putText(img, '{:.2f}'.format(score), (x1, y1 - 6)
								, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (22, 22, 250), 2)

			# Add keypoints for the current face
			for keypoint in keypoints:
				xKeypoint = (keypoint[0] * self.img_width).astype(int)
				yKeypoint = (keypoint[1] * self.img_height).astype(int)
				cv2.circle(img,(xKeypoint,yKeypoint), 4, (214, 202, 18), -1)

		# Add fps counter
		# cv2.putText(img, f'FPS: {self.fps}', (40, 40)
		# 						,cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 250, 22), 2)

		return img

	def getModelInputDetails(self):
		self.input_details = self.interpreter.get_input_details()
		input_shape = self.input_details[0]['shape']
		self.inputHeight = input_shape[1]
		self.inputWidth = input_shape[2]
		self.channels = input_shape[3]

	def getModelOutputDetails(self):
		self.output_details = self.interpreter.get_output_details()

	def generateAnchors(self, type):
		if type == "front":
			# Options to generate anchors for SSD object detection models.
			ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=128, input_size_height=128, min_scale=0.1484375, max_scale=0.75
					, anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4
					, feature_map_width=[], feature_map_height=[]
					, strides=[8, 16, 16, 16], aspect_ratios=[1.0]
					, reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0
					, fixed_anchor_size=True)

		elif type == "back":
			# Options to generate anchors for SSD object detection models.
			ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=256, input_size_height=256, min_scale=0.15625, max_scale=0.75
					, anchor_offset_x=0.5, anchor_offset_y=0.5, num_layers=4
					, feature_map_width=[], feature_map_height=[]
					, strides=[16, 32, 32, 32], aspect_ratios=[1.0]
					, reduce_boxes_in_lowest_layer=False, interpolated_scale_aspect_ratio=1.0
					, fixed_anchor_size=True)

		self.anchors = gen_anchors(ssd_anchors_calculator_options)

	def prepareInputForInference(self, image):
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.img_height, self.img_width, self.img_channels = img.shape

		# Input values should be from -1 to 1 with a size of 128 x 128 pixels for the fornt model
		# and 256 x 256 pixels for the back model
		img = img / 255.0
		img_resized = tf.image.resize(img, [self.inputHeight,self.inputWidth], 
									method='bicubic', preserve_aspect_ratio=False)
		img_input = img_resized.numpy()
		img_input = (img_input - 0.5) / 0.5

		# Adjust matrix dimenstions
		reshape_img = img_input.reshape(1,self.inputHeight,self.inputWidth,self.channels)
		tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

		return tensor

	def inference(self, input_tensor):
		self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
		self.interpreter.invoke()

		# Matrix of 896 x 16 with information about the detected faces
		output0 = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))

		# Matrix with the raw detection scores
		output1 = np.squeeze(self.interpreter.get_tensor(self.output_details[1]['index']))

		return output0, output1

	def extractDetections(self, output0, goodDetectionsIndices):

		numGoodDetections = goodDetectionsIndices.shape[0]

		keypoints = np.zeros((numGoodDetections, KEY_POINT_SIZE, 2))
		boxes = np.zeros((numGoodDetections, 4))
		for idx, detectionIdx in enumerate(goodDetectionsIndices):
			anchor = self.anchors[detectionIdx]

			sx = output0[detectionIdx, 0]
			sy = output0[detectionIdx, 1]
			w = output0[detectionIdx, 2]
			h = output0[detectionIdx, 3]

			cx = sx + anchor.x_center * self.inputWidth
			cy = sy + anchor.y_center * self.inputHeight

			cx /= self.inputWidth
			cy /= self.inputHeight
			w /= self.inputWidth
			h /= self.inputHeight

			for j in range(KEY_POINT_SIZE):
				lx = output0[detectionIdx, 4 + (2 * j) + 0]
				ly = output0[detectionIdx, 4 + (2 * j) + 1]
				lx += anchor.x_center * self.inputWidth
				ly += anchor.y_center * self.inputHeight
				lx /= self.inputWidth
				ly /= self.inputHeight
				keypoints[idx,j,:] = np.array([lx, ly])

			boxes[idx,:] = np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])

		return boxes, keypoints

	def filterDetections(self, output1):

		# Filter based on the score threshold before applying sigmoid function
		goodDetections = np.where(output1 > self.sigmoidScoreThreshold)[0]

		# Convert scores back from sigmoid values
		scores = 1.0 /(1.0 + np.exp(-output1[goodDetections]))

		return scores, goodDetections

	def filterWithNonMaxSupression(self, boxes, keypoints, scores):
		# Filter based on non max suppression
		selected_indices = tf.image.non_max_suppression(boxes, scores, MAX_FACE_NUM, self.iouThreshold)
		filtered_boxes = tf.gather(boxes, selected_indices).numpy()
		filtered_keypoints = tf.gather(keypoints, selected_indices).numpy()
		filtered_scores = tf.gather(scores, selected_indices).numpy()

		detectionResults = Results(filtered_boxes, filtered_keypoints, filtered_scores)
		return detectionResults

class Results:
	def __init__(self, boxes, keypoints, scores):
		self.boxes = boxes
		self.keypoints = keypoints
		self.scores = scores
