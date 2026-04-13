import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==============================================================================
# 1. THE EXACT KAGGLE ARCHITECTURE
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # FIX 1: bias=False to match the saved Kaggle weights
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# FIX: Changed ratio=16 to ratio=8 to match the Kaggle checkpoint dimensions
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8): 
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

# FIX: Changed ratio=16 to ratio=8 here as well
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class VeinHunterUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(VeinHunterUNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        
        self.upConv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(256, 128) 
        self.cbam3 = CBAM(128) 
        
        self.upConv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = DoubleConv(128, 64)
        self.cbam2 = CBAM(64)  
        
        self.upConv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up1 = DoubleConv(64, 32)
        self.cbam1 = CBAM(32)  
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        b = self.bottleneck(self.pool(x3))
        
        x = self.upConv3(b)
        x = torch.cat([x, x3], dim=1) 
        x = self.up3(x)
        x = self.cbam3(x)             
        
        x = self.upConv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.cbam2(x)             
        
        x = self.upConv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = self.cbam1(x)             
        
        logits = self.out_conv(x)
        return self.sigmoid(logits)

# ==============================================================================
# 2. THE INFERENCE FUNCTION (PURE RGB)
# ==============================================================================
def test_random_image_rgb(image_path, weights_path='vein_hunter_FINAL_SKIN_weights.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    # Load the Model
    model = VeinHunterUNet().to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("[*] Weights loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading weights: {e}")
        return
        
    model.eval() # Freezes Dropout and BatchNorm

    # Load the Raw Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[!] Error: Could not load image at {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize EXACTLY as done in standard training
    pil_raw_rgb = Image.fromarray(img_rgb).resize((256, 256)) 
    
    # Convert to PyTorch Tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = transform(pil_raw_rgb).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        prediction = model(input_tensor)
        pred_prob = prediction.squeeze().cpu().numpy()
        
    # Strict thresholding
    binary_mask = (pred_prob > 0.5).astype(np.uint8)

    # Overlay
    overlay = np.array(pil_raw_rgb).copy()
    overlay[binary_mask == 1] = [0, 255, 0]

    # Plot everything side-by-side
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(pil_raw_rgb)
    ax[0].set_title("1. Original RGB (Model Input)")
    ax[0].axis('off')

    ax[1].imshow(binary_mask, cmap='gray')
    ax[1].set_title(f"2. AI Mask (Max Conf: {pred_prob.max():.2f})")
    ax[1].axis('off')

    ax[2].imshow(overlay)
    ax[2].set_title("3. Robotic Targeting Overlay")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. RUN IT
# ==============================================================================
if __name__ == "__main__":
    IMAGE_TO_TEST = 'hand.jpeg' # <--- CHANGE THIS TO YOUR IMAGE NAME
    test_random_image_rgb(IMAGE_TO_TEST)