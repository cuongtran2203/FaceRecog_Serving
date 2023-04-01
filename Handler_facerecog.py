import torch
import cv2
import onnxruntime as ort
import numpy as np
def to_input(img):
    brg_img = ((img / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    return tensor

