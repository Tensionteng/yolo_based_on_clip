from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model_origin = YOLO('model/yolov8m-obb.pt')
model_train = YOLO('runs/obb/train5/weights/best.pt')

# Define path to the image file
source = "datasets/DOTAv1/images/test/P0014.jpg"

# Run inference on the source
result_1 = model_origin(source, save=True)  # list of Results objects
result_2 = model_train(source, save=True)
