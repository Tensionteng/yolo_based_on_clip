from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model_origin = YOLO('models/yolov8m-obb.pt')
model_train = YOLO('runs/obb/train2/weights/best.pt')

# Define path to the image file
source = "datasets/DOTAv1/images/test/P2799.jpg"

# Run inference on the source
result_1 = model_origin(source, save=True)  # list of Results objects
result_2 = model_train(source, save=True)

# from ultralytics import YOLO

# # Load a model
# model = YOLO('models/yolov8m-obb.pt')

# # Validate the model
# metrics = model.val(data="DOTAv1.yaml")  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category