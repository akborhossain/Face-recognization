import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms




# Define MiniFASNetV2 model architecture
class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=2, input_size=(80, 80)):
        super(MiniFASNetV2, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

class AntiSpoofPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MiniFASNetV2()
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
        ])

    def predict(self, face_img):
        img = self.transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
            prob = F.softmax(output, dim=1)
        label = torch.argmax(prob, dim=1).item()
        return label, prob[0][label].item()
