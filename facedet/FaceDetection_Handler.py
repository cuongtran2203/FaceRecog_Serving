import torch
import onnxruntime as ort 
import numpy as np
import cv2
from preprocess import *
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import os
import io
class FaceDetectionHandler(BaseHandler):
    def __init__(self) :
        self._context=None
        self.initialized=False
        self.providers = ['CPUExecutionProvider']
        self.session =None
        self.data_input=None
        self.cfg_mnet = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False
                }
    def initialize(self, context):
        self._context=context
        self.manifest=context.manifest
        properties=context.system_properties
        model_dir = properties.get("model_dir")
        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.session=ort.InferenceSession(model_pt_path, providers=self.providers)
        self.initialized=True
    def preprocess(self,data):
        image=data[0].get("data")
        if image is None:
            image= data[0].get("body")
        image = Image.open(io.BytesIO(image))
        img=np.array(image)
        self.data_input=img
        img=cv2.resize(img,(640,640))
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        return img.unsqueeze(0)
    def inference(self,blob):
        outputs = self.session.run(None, {self.session.get_inputs()[0].name:np.asarray(blob)})
        return outputs
    def postprocess(self,preds):
        loc, conf, landms=preds
        conf=torch.as_tensor(conf)
        scale = torch.Tensor([self.data_input.shape[1], self.data_input.shape[0],self.data_input.shape[1], self.data_input.shape[0]])
        loc=torch.as_tensor(loc).reshape(1,16800,4)
        landms=torch.as_tensor(landms).reshape(1,16800,10)
        priorbox = PriorBox(self.cfg_mnet, image_size=(640,640))
        priors = priorbox.forward()
        prior_data = priors.data
        resize=1
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_mnet['variance'])
        boxes = boxes * scale / resize
        scale1 = torch.Tensor([self.data_input.shape[1], self.data_input.shape[0], self.data_input.shape[1], self.data_input.shape[0],
                        self.data_input.shape[1], self.data_input.shape[0], self.data_input.shape[1], self.data_input.shape[0],
                        self.data_input.shape[1], self.data_input.shape[0]])
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.4)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        #print(order)
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:5000, :]
        landms = landms[:5000, :]
        dets = np.concatenate((dets, landms), axis=1)
        bbox=[]
        res={}
        for b in dets:
            if b[4] < 0.6:
                continue
            b = list(map(int, b))
            x,y,w,h=int(b[0]),int(b[1]),int(b[2]),int(b[3])
            bbox.append([x,y,w,h])
            res["bbox"]=bbox
        return res
    def handle(self, data, context):
        if not self.initialized:
            self.initialized(context)
        model_input=self.preprocess(data)
        model_output=self.inference(model_input)
        return self.postprocess(model_output)
