import os
import cv2
import time
import threading
from imutils.video import VideoStream
from akida import Model as AkidaModel, devices
from akida_models import yolo_widerface_pretrained, yolo_base
from akida_models.detection.processing import (
    decode_output,
)
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Reshape
from cnn2snn import convert
import numpy as np


CAMERA_SRC = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

INFERENCE_PER_SECOND = 2
PREDICTION_CONFIDENCE_MIN = 0.6

TARGET_WIDTH = 224
TARGET_HEIGHT = 224

MODEL_FBZ = "models/yolo_face.fbz"
NUM_ANCHORS = 3
GRID_SIZE = (7, 7)
NUM_CLASSES = 1


def initialise():

    # Create a yolo model for 2 classes with 10 anchors and grid size of 7
    model = yolo_base(
        input_shape=(TARGET_WIDTH, TARGET_HEIGHT, 3),
        classes=NUM_CLASSES,
        nb_box=NUM_ANCHORS,
        alpha=0.5,
    )

    # Define a reshape output to be added to the YOLO model
    output = Reshape(
        (GRID_SIZE[1], GRID_SIZE[0], NUM_ANCHORS, 6),
        name="YOLO_output",
    )(model.output)

    # Build the complete model
    full_model = Model(model.input, output)
    full_model.output

    # Load the pretrained model along with anchors
    model_keras, anchors = yolo_widerface_pretrained()

    # Define the final reshape and build the model
    output = Reshape(
        (GRID_SIZE[1], GRID_SIZE[0], NUM_ANCHORS, 6),
        name="YOLO_output",
    )(model_keras.output)
    model_keras = Model(model_keras.input, output)

    # Rebuild a model without the last layer
    compatible_model = Model(model_keras.input, model_keras.layers[-2].output)
    model_akida = convert(compatible_model)
    model_akida.summary()

    # save the model to file
    model_akida.save(MODEL_FBZ)


# create a model if one doesnt exist
if not os.path.exists(MODEL_FBZ):
    print("Initialising Akida model")
    initialise()


"""
Class to capture video feed from webcam
"""


class Camera:
    def __init__(self):
        self.stream = VideoStream(
            src=CAMERA_SRC, resolution=(FRAME_WIDTH, FRAME_HEIGHT)
        ).start()
        self.label = 0
        self.pred_boxes = []

    def get_frame(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        return frame

    def get_input_array(self):
        frame = cv2.resize(self.stream.read(), (TARGET_WIDTH, TARGET_HEIGHT))
        input_array = img_to_array(frame)
        input_array = np.array([input_array], dtype="uint8")
        return input_array

    def show_frame(self):
        frame = self.render_boxes(self.stream.read())
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

    def render_boxes(self, frame):
        for box in self.pred_boxes:
            if box[5] > PREDICTION_CONFIDENCE_MIN:
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(box[2]), int(box[3])
                label = "FACE"
                score = "{:.2%}".format(box[5])
                colour = (255, 255, 255)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
                cv2.putText(
                    frame,
                    "{} - {}".format(label, score),
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour,
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def set_prod_boxes(self, pred_boxes):
        self.pred_boxes = pred_boxes


"""
Class to run inference over frames from the webcam
"""


class Inference:
    def __init__(self, camera):
        # init the camera
        self.camera = camera
        _, self.anchors = yolo_widerface_pretrained()
        # run inference in separate thread
        self.t1 = threading.Thread(target=self.infer)
        self.t1.start()
        # load the akida model
        self.model_ak = AkidaModel(filename=MODEL_FBZ)

        if len(devices()) > 0:
            device = devices()[0]
            self.model_ak.map(device)

    def infer(self):
        while True:

            input_array = self.camera.get_input_array()

            # Call evaluate on the image
            pots = self.model_ak.predict(input_array)[0]

            # print(pots.shape)

            # time.sleep(1000)

            w, h, c = pots.shape
            pots = pots.reshape((h, w, len(self.anchors), 6))

            # print(pots)

            # Decode potentials into bounding boxes
            raw_boxes = decode_output(pots, self.anchors, 1)

            # Rescale boxes to the original image size
            pred_boxes = np.array(
                [
                    [
                        box.x1 * FRAME_WIDTH,
                        box.y1 * FRAME_HEIGHT,
                        box.x2 * FRAME_WIDTH,
                        box.y2 * FRAME_HEIGHT,
                        box.get_label(),
                        box.get_score(),
                    ]
                    for box in raw_boxes
                ]
            )

            self.camera.set_prod_boxes(pred_boxes)

            time.sleep(1 / INFERENCE_PER_SECOND)


camera = Camera()
inference = Inference(camera)

# main loop to display camera feed
while True:
    camera.show_frame()
