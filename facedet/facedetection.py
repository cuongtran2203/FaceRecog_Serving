import torch
import onnxruntime as onnxrt
import cv2
import torch
import numpy as np
from preprocess import *
import cv2
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}
device = "cpu"
img_path="test.jpg"
img=cv2.imread(img_path)
#preprocess image

img=cv2.resize(img,(640,640))
im=img.copy()
img = np.float32(img)
im_height, im_width, _ = img.shape
scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
img -= (104, 117, 123)
img = img.transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0)
img = img.to(device)
scale = scale.to(device)
onnx_session = onnxrt.InferenceSession("FaceDetector_resnet50.onnx")
onnx_input = {onnx_session.get_inputs()[0].name:img.cpu().detach().numpy()}
loc, conf, landms= onnx_session.run(None,onnx_input)
conf=torch.as_tensor(conf)
#print(loc.shape)
print(type(loc))
loc=torch.as_tensor(loc).reshape(1,16800,4)

print("LOC SHAPE :",loc.shape)

landms=torch.as_tensor(landms).reshape(1,16800,10)
print("lamds shape :",landms.shape)
priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
priors = priorbox.forward()
priors = priors.to(device)
prior_data = priors.data
#print(priors.shape)
resize=1
boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
boxes = boxes * scale / resize
boxes = boxes.cpu().numpy()
scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2]])
scale1 = scale1.to(device)
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
print("dets shape ",dets.shape)
if save_image:
            for b in dets:
                if b[4] < 0.6:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                x,y,w,h=int(b[0]),int(b[1]),int(b[2]),int(b[3])
                cv2.rectangle(im, (x, y), (w, h), (0, 0, 255), 2)
                cx = x
                cy = y + 12
                cv2.putText(im, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(im, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(im, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(im, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(im, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(im, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test123.jpg"
            cv2.imwrite(name, im)

