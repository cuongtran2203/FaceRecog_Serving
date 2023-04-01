import torch 
import cv2
from PIL import Image
import io
import onnxruntime
from preprocess import *
from ts.torch_handler.base_handler import BaseHandler
class FaceDetector_Handler(BaseHandler):
    def __init__(self):
        self.context=None
        self.initialized=False
        self.cfg_mnet = {
            
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False 
        }

        self.model_path="FaceDetecor.onnx"
        self.providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
    def initialize(self, context):
        self.context=context
        self.manifest=context.manifest
        properties=context.system_properties
        self.initialized=True
    def preprocess(self,data):
        image=data[0].get("data")
        
        if image is None:
            image= data[0].get("body")
        image = Image.open(io.BytesIO(image))
        
        img=np.array(image)
        img_re=cv2.resize(img,(640,640))
        img_re=np.float32(img_re)
        img_re -= (104, 117, 123)
        img_re = img_re.transpose(2, 0, 1)
        return np.expand_dims(img_re, axis=0)
    def inference(self,blob):
        outputs = self.session.run(None, {self.session.get_inputs()[0].name:np.asarray(blob)})
        # print("time :{:.3f} s".format(time.perf_counter()-start))
        return outputs
    def postprocess(self,preds):
        
        loc,conf,landms=preds
        conf=torch.as_tensor(conf)
        #print(loc.shape)
        print(type(loc))
        loc=torch.as_tensor(loc).reshape(1,16800,4)

        print("LOC SHAPE :",loc.shape)

        landms=torch.as_tensor(landms).reshape(1,16800,10)
        print("lamds shape :",landms.shape)
        priorbox = PriorBox(self.cfg_mnet, image_size=(640,640))
        priors = priorbox.forward()
        prior_data = priors.data
        resize=1
        scale=torch.Tensor([640,640,640,640])
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data,self.cfg_mnet['variance'])
        scale1 = torch.Tensor([640, 640, 640, 640,
                                640, 640, 640, 640,
                                640, 640])
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
        save_image=True
        dets = np.concatenate((dets, landms), axis=1)
        bbox=[]
        res={}
        for b in dets :
            if b[4] < 0.6:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            bbox.append(b)
        res["bbox"]=bbox  
        return [res]

        
        