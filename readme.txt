Để giảm kích thước, em đã xóa file detectron2 do đó frcnn cần cài lại detectron2 để có thể chạy được
python -m pip install pyyaml
git clone 'https://github.com/facebookresearch/detectron2'
----------------------
file train mode với GPU:
-- trainFRCNN.ipynp : train model với Faster RCNN
-- trainYOLO.ipynp : train model với YOLOv8m
file test local:
-- frcnn.py : chạy test với model train lại với Faster RCNN
-- yolov8m.py : chạy test với yolo
-- streamlit_app.py : chạy test yolo với giao diện streamlit