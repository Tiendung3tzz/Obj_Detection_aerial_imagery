import cv2
import os
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo

# Cấu hình Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("frcnnw\model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Thiết lập ngưỡng tùy chỉnh cho việc test
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Số lượng lớp của bạn, cần chỉnh sửa cho phù hợp với mô hình của bạn

# Thiết lập để sử dụng CPU
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

# Tạo thư mục để lưu kết quả
Path("result3").mkdir(parents=True, exist_ok=True)

# Mở video
video_path = 'video/uk1.mp4'
cap = cv2.VideoCapture(video_path)

# Lấy thông tin về video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Tạo đối tượng để ghi video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

frame_id = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Xử lý từng khung hình
    outputs = predictor(frame)
    v = Visualizer(frame[:, :, ::-1], scale=1)
    out_frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_frame = out_frame.get_image()[:, :, ::-1]

    # Ghi khung hình đã xử lý vào video
    out.write(out_frame)
    frame_id += 1
    print(f"Processed frame {frame_id}/{total_frames}")

cap.release()
out.release()
cv2.destroyAllWindows()
