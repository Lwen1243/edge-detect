import cv2
import numpy as np
import torch

def detect_table_edges_with_yolov5():
    # 1. 加载YOLOv5模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头是否正常连接。")
        return

    while True:
        # 3. 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧。")
            break

        # 4. 使用YOLOv5进行物体检测
        results = model(frame)

        # 5. 提取桌子的检测结果
        tables = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'dining table']

        # 6. 在原图上绘制桌子边缘
        for table in tables.itertuples():
            x1, y1, x2, y2 = int(table.xmin), int(table.ymin), int(table.xmax), int(table.ymax)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 7. 显示结果
        cv2.imshow("Table Edges", frame)

        # 8. 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. 释放摄像头资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

# 运行实时检测
detect_table_edges_with_yolov5()