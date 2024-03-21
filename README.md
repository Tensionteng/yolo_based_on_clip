fork from [yolov8](https://github.com/ultralytics/ultralytics)

# usage
```
# train
python main.py

# predict
python predict.py
```

# dataset
- [DOTAv1](https://github.com/ultralytics/yolov5/releases/download/v1.0/DOTAv1.zip)


# Architecture
- [yolov8m-obb-clip.yaml](https://github.com/Tensionteng/yolo_based_on_clip/blob/main/yolov8m-obb-clip.yaml)

```yaml
# Parameters
nc: 15 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Clip, [["traffic"], True]] # 0
  - [-1, 1, Clip, [["traffic"], False]] # 1
  - [ 0, 1, Conv, [64, 3, 2]] # 2-P1
  - [-1, 1, Conv, [128, 3, 2]] # 3-P2
  - [-1, 3, C2f, [128, True]]  # 4
  - [-1, 1, Conv, [256, 3, 2]] # 5-P3
  - [-1, 6, C2f, [256, True]]  # 6
  - [-1, 1, Conv, [512, 3, 2]] # 7-P4
  - [-1, 6, C2f, [512, True]]  # 8
  - [-1, 1, Conv, [1024, 3, 2]] # 9-P5
  - [-1, 3, C2f, [1024, True]] # 10
  - [-1, 1, SPPF, [1024, 5]] # 11

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] #12
  - [[-1, 8], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 14

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 17 (P3/8-small)
  - [[-1, 1], 1, CrossAttention, [7]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 21 (P4/16-medium)
  - [[-1, 1], 1, CrossAttention, [7]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 25 (P5/32-large)
  - [[-1, 1], 1, CrossAttention, [7]]

  - [[18, 22, 26], 1, OBB, [nc, 1]] # OBB(P3, P4, P5)
```

# New Block
## CrossAttention
```python
class CrossAttention(nn.Module):
    def __init__(self, channel: int, clip_channel: int, kernel_size: int = 7):
        super(CrossAttention, self).__init__()
        self.cross_attention = CBAM(c1=channel, kernel_size=kernel_size)
        

    def forward(self, x):
        x, clip_feature = x[0], x[1]
        self.device = x.device
        _, _, h, w = x.size()
        batch, in_channel = clip_feature.size()

        mlp = nn.Sequential(
            nn.Linear(in_channel, h * w),
            nn.ReLU(),
        ).to(self.device)

        x = x.to(self.device)
        residual = x
        with autocast():
            clip_feature = mlp(clip_feature)
        clip_feature = (
            torch.reshape(clip_feature, (batch, h, w)).unsqueeze(1).expand_as(x)
        )

        x = self.cross_attention(x + clip_feature)
        return residual + x

```
## Clip
```python
class Clip(nn.Module):
    def __init__(
        self, channel, text: list[str] = ["traffic"], skip: bool = True
    ) -> None:
        super(Clip, self).__init__()
        self.model, _ = clip.load("ViT-B/32")
        # self.model.float()
        self.model.eval()
        self.text = text
        self.skip = skip

    def forward(self, x: torch.Tensor):
        if self.skip:
            return x

        self.device = x.device
        if str(self.device) == "cpu":
            self.model.float()

        self.model = self.model.to(self.device)

        with torch.no_grad():
            with autocast():
                if self.training:
                    import torchvision.transforms as transforms

                    clip_preprocess = transforms.Compose(
                        [
                            transforms.Resize(
                                (224, 224),
                                antialias=True,
                            ),
                            transforms.Normalize(
                                (0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711),
                            ),
                        ]
                    )
                    x = clip_preprocess(x).to(self.device)
                    features = self.model.encode_image(x)
                else:
                    text_input = clip.tokenize(self.text).to(self.device)
                    features = self.model.encode_text(text_input)

        return features

```