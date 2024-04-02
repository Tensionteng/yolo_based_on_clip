from ultralytics import YOLO

def train_DOTAv1_5():
    model = YOLO("models/yolov8m-obb.pt")
    results = model.train(data="./DOTAv1.5.yaml", epochs=100, imgsz=640, batch=16, freeze=20)

def train_DOTAv2():
    # model = YOLO("yolov8-obb.yaml")
    # model.load("models/yolov8m-obb.pt")
    model = YOLO("models/yolov8m-obb.pt")
    results = model.train(data="./DOTAv2.yaml", epochs=100, imgsz=640, batch=16)

if __name__ == "__main__":
    train_DOTAv1_5()
    # train_DOTAv2()
