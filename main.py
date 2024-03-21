from ultralytics import YOLO

model = YOLO("yolov8m-obb-clip.yaml")
pretrain_model = YOLO("model/yolov8m-obb.pt")


def get_num(layer: str) -> int:
    try:
        layer_num = int(layer[12:14])
    except:
        layer_num = int(layer[12:13])
        if layer_num == 0 or layer_num == 1:
            return None

    if layer_num >= 19 and layer_num <= 21:
        layer_num -= 1
    elif layer_num >= 23 and layer_num <= 25:
        layer_num -= 2
    elif layer_num >= 27:
        layer_num -= 3
    return layer_num - 2


pretrain_model = pretrain_model.state_dict()

# load pre train model weights
for k, v in model.named_parameters():
    if get_num(k) is None:
        v.requires_grad = False
        continue
    layer_num = get_num(k)
    name = (
        f"{k[0:12]}{layer_num}{k[14:]}"
        if layer_num >= 8
        else f"{k[0:12]}{layer_num}{k[13:]}"
    )
    if name in pretrain_model:
        v.requires_grad = False
        v.copy_(pretrain_model[name])

# freeze the model
# def freeze_layer(trainer):
#     model = trainer.model
#     num_freeze = [
#         0,
#         1,
#         2,
#         3,
#         4,
#         5,
#         6,
#         7,
#         8,
#         9,
#         10,
#         11,
#         12,
#         13,
#         14,
#         15,
#         16,
#         17,
#         19,
#         20,
#         21,
#         23,
#         24,
#         25,
#     ]
#     print(f"Freezing {num_freeze} layers")
#     freeze = [f'model.{x}.' for x in num_freeze]
#     for k, v in model.named_parameters():
#         v.requires_grad = True  # train all layers
#         if any(x in k for x in freeze):
#             print(f'freezing {k}')
#             v.requires_grad = False
#     print(f"{num_freeze} layers are freezed.")

# model.add_callback("on_train_start", freeze_layer)

results = model.train(data="./DOTAv1.yaml", epochs=100, imgsz=640, batch=16)
