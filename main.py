import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# VGG16 をフルスクラッチで定義
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# デバイスの設定（GPUがあればGPUを使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのインスタンス化と保存済みパラメータのロード
model = VGG16(num_classes=10).to(device)
model.load_state_dict(torch.load("vgg16_scratch_cifar10_updated.pth", map_location=device))
model.eval()

# CIFAR-10 の各クラス（推論結果を見やすくするため）
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ユーザー入力画像の前処理（モデル学習時と同じ設定）
transform = transforms.Compose([
    transforms.Resize((32, 32)),         # CIFAR-10 のサイズに合わせる
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                         std=[0.5, 0.5, 0.5])
])

# Streamlit アプリのUI
st.title("VGG16 CIFAR-10 推論アプリ")
st.write("アップロードした画像を CIFAR-10 のクラスに分類します。")

uploaded_file = st.file_uploader("画像ファイルを選択してください (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_column_width=True)
    
    # 前処理してモデル入力用テンソルへ変換
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推論処理
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    st.write(f"推論結果: **{classes[predicted.item()]}**")
