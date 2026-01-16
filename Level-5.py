import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Constants
# -----------------------
IMAGE_SIZE = 224
NUM_CLASSES = 102

# -----------------------
# Class names
# -----------------------
FLOWER_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata",
    "japanese anemone", "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine",
    "desert rose", "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum", "bee balm",
    "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]

# -----------------------
# Image preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# CBAM Module
# -----------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * ca
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x * sa

# -----------------------
# Attention ResNet-50
# -----------------------
class AttentionResNet50(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        base = models.resnet50(pretrained=False)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.cbam1 = CBAM(256)
        self.layer2 = base.layer2
        self.cbam2 = CBAM(512)
        self.layer3 = base.layer3
        self.cbam3 = CBAM(1024)
        self.layer4 = base.layer4
        self.cbam4 = CBAM(2048)
        self.pool = base.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# -----------------------
# Load models
# -----------------------
model_l2 = models.resnet50(pretrained=False)
model_l2.fc = nn.Linear(model_l2.fc.in_features, NUM_CLASSES)
model_l2.load_state_dict(torch.load("level2_resnet50_aug_new_with_layer_3.pth", map_location=device))
model_l2.to(device).eval()

model_l3 = AttentionResNet50(NUM_CLASSES)
model_l3.load_state_dict(torch.load("level3_attention_resnet50_cbam.pth", map_location=device))
model_l3.to(device).eval()

# -----------------------
# Prediction function
# -----------------------
def predict(image):
    image = image.convert("RGB")
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = F.softmax(model_l2(img), dim=1)
        p2 = F.softmax(model_l3(img), dim=1)
        probs = 0.5 * p1 + 0.5 * p2

    probs = probs.cpu().numpy()[0]
    top_idx = probs.argmax()
    return {
        FLOWER_CLASSES[top_idx]: float(probs[top_idx])
    }

# -----------------------
# Gradio Interface
# -----------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Flower Image"),
    outputs=gr.Label(num_top_classes=3),
    title="Flowers-102 Ensemble Classifier",
    description="Level-5 deployment using a soft-voting ensemble of ResNet-50 and CBAM-ResNet."
)

if __name__ == "__main__":
    demo.launch()
