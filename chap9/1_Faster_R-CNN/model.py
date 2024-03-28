from torchvision.models import vgg16
from torchvision import ops
from torchvision.models.detection import rpn
from torchvision.models.detection import FasterRCNN

# 백본과 출력 채널 작성
backbone = vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features
backbone.out_channels = 512

# 앵커 박스
anchor_generator = rpn.AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),), # 콤마(,)는 하나의 요소만 갖는 튜플의 명시적 표현
    aspect_ratios=(0.5, 1.0, 2.0) # 종횡비
)
roi_pooler = ops.MultiScaleRoIAlign( # 관심영역 풀링
    featmap_names=["0"], # VGG-16 모델의 특징 추출 계층 이름은 "0"
    output_size=(7, 7), # 특징맵의 크기
    sampling_ratio=2 # 2x2 그리드, 원본 특징 맵 영역을 샘플링
)
model = FasterRCNN(
    backbone=backbone,
    num_classes=3, # 배경 + 클래스 2개
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

# 2