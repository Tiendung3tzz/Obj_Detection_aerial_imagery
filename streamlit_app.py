import cv2
from sort import *
import streamlit as st
from ultralytics import YOLO
import cvzone
import math
from PIL import Image
import numpy as np
import tempfile

# Định nghĩa các tên lớp
classNames = ['Soldier', 'Tank', 'Truck_army']
color ={
                0:(255,0,0),
                1:(0,255,0),
                2:(0,0,255),
             }
# Khởi tạo bộ theo dõi
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Đặt giới hạn phát hiện
limits = [300, 400, 673, 297]

# Giao diện Streamlit
st.title("Phát Hiện và Theo Dõi Đối Tượng Theo Thời Gian Thực")

# Thanh trượt ngưỡng độ tin cậy
conf_threshold = st.sidebar.slider("Ngưỡng Độ Tin Cậy", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

# Thanh trượt bỏ qua khung hình
frame_skip = st.sidebar.slider("Bỏ Qua Khung Hình", min_value=1, max_value=10, value=1, step=1)

# Chọn loại đầu vào
input_type = st.radio("Chọn loại đầu vào", ("Ảnh", "Video"))

# Kiểm tra và tải mô hình YOLO
if 'model' not in st.session_state:
    st.session_state.model = YOLO('yolo_weights/last3.pt')

model = st.session_state.model

# Tải lên tệp
if input_type == "Ảnh":
    uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.file_uploader("Chọn một video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    if input_type == "Ảnh":
        # Đọc hình ảnh đã tải lên
        img = np.array(Image.open(uploaded_file))
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Chỉ lấy 3 kênh đầu tiên (RGB)

        # Thay đổi kích thước hình ảnh
        height, width = img.shape[:2]
        new_width = int(width)
        new_height = int(height)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Phát hiện đối tượng
        if st.button("Phát Hiện"):
            # Khởi tạo mảng phát hiện
            detections = np.empty((0, 5))

            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    bbox = x1, y1, w, h

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    labelDec = ['Soldier', 'Tank', 'Truck_army']

                    if (currentClass in labelDec) and (conf > conf_threshold):
                        cvzone.cornerRect(img, bbox, l=8, rt=5,colorR=color[cls])
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))
                        cvzone.putTextRect(
                            img,
                            f'{classNames[cls]} - {conf}',
                            (max(0, x1), max(35, y1 - 20)),
                            scale=1,
                            thickness=2,
                            offset=7
                        )

            # Cập nhật bộ theo dõi với các phát hiện
            resultsTracker = tracker.update(detections)

            # Chuyển đổi hình ảnh sang định dạng RGB cho Streamlit
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption='Hình ảnh đã xử lý', use_column_width=True)

    elif input_type == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        
        # Định nghĩa bộ giải mã video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(out_file.name, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        frame_id = 0
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            if frame_id % frame_skip == 0:
                # Thay đổi kích thước hình ảnh
                height, width = img.shape[:2]
                new_width = int(width)
                new_height = int(height)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Khởi tạo mảng phát hiện
                detections = np.empty((0, 5))

                results = model(img, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        bbox = x1, y1, w, h

                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]
                        labelDec = ['Soldier', 'Tank', 'Truck_army']

                        if (currentClass in labelDec) and (conf > conf_threshold):
                            cvzone.putTextRect(
                                img,
                                f'{classNames[cls]} - {conf}',
                                (max(0, x1), max(35, y1 - 20)),
                                scale=1,
                                thickness=2,
                                offset=7
                            )
                            cvzone.cornerRect(img, bbox, l=8, rt=5,colorR=color[cls])
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            detections = np.vstack((detections, currentArray))

                # Cập nhật bộ theo dõi với các phát hiện
                resultsTracker = tracker.update(detections)

                # Chuyển đổi hình ảnh sang định dạng RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption='Hình ảnh đã xử lý', use_column_width=True)
                # Viết khung đã xử lý vào video output
                out.write(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            frame_id += 1

       

