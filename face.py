import cv2

# 加载YOLOv5模型
model_weights = "yolov5s.pt"
model_cfg = "models/yolov5s.yaml"
net = cv2.dnn.readNet(model_weights, model_cfg)

# 设置输入和输出节点名称
input_name = "images"
output_names = ["output"]

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 获取一帧图像
    ret, frame = cap.read()

    # 对图像进行预处理和缩放
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1) / 255.0
    img = img.reshape(1, 3, 640, 640)

    # 将图像输入到模型中进行目标检测
    net.setInput(img)
    outputs = net.forward(output_names)

    # 解析模型输出结果并绘制检测框和标签
    for output in outputs:
        for detection in output:
            conf = detection[4]
            if conf > 0.5:
                x, y, w, h = detection[:4] * 640
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                label = detection[5]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果图像
    cv2.imshow("YOLOv5 Real-time Detection", frame)

    # 按下q键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
