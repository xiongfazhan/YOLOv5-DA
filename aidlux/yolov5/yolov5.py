from cvs import *
import aidlite_gpu
from utils import detect_postprocess, preprocess_img, draw_detect_res

#加载模型
model_path = 'best-fp16.tflite'
in_shape = [1 * 640 * 640 * 3 * 4, ]
out_shape = [1*25200*85*4, 1*3*80*80*85*4, 1*3*40*40*85*4, 1*3*20*20*85*4]

aidlite = aidlite_gpu.aidlite()
aidlite.ANNModel(model_path,in_shape,out_shape,4,0)

cap = cvs.VideoCapture(0)

while True:
    frame = cap.read()
    if frame is None:
        continue

    #预处理
    img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None,stds=None)
    
    aidlite.setInput_Float32(img,640,640)

    #推理
    aidlite.invoke()

    pred = aidlite.getOutput_Float32(0)
    pred = pred.reshape(1,25200,85)[0]
    pred = detect_postprocess(pred, frame.shape, [640, 640, 3], conf_thres=0.5,iou_thres=0.45)
    res_img = draw_detect_res(frame, pred)
    cvs.imshow(res_img)


