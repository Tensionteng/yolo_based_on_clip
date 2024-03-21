from ultralytics import YOLO

CFG = "yolov8m-obb-clip.yaml"


def test_model_forward():
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment
