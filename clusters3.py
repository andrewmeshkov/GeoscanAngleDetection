
import cv2
import time
import numpy as np
from roboflow import Roboflow
import math as m

class CVAdapter:
    def __init__(self, api_key, model, version, confidence):
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(model)
        self.model = project.version(version).model

        self.confidence = confidence

    def make_prediction(self, img):
        return self.model.predict(img, confidence=self.confidence, overlap=30).json()

    def angle(self, img):
        preds = self.make_prediction(img)
        return -m.degrees(m.atan(preds['predictions'][0]['height'] / preds['predictions'][0]['width']))