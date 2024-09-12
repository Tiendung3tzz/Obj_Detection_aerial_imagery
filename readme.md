#evaluation
![image](https://github.com/user-attachments/assets/dcdb983d-1243-41b8-8711-f8efd301f5d5)
![image](https://github.com/user-attachments/assets/c1fccf87-d6cb-4fc2-9c69-47638593d97f)

#dataset
bộ data được thu thập từ những video drone trên chiến trường
kết quả thực nghiệm
![image](https://github.com/user-attachments/assets/3380a4fd-b934-4541-ae52-45f1e4e923d1)

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
