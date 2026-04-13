import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==============================================================================
# 1. THE EXACT KAGGLE ARCHITECTURE (WITH CBAM RATIO=8)
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

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
# 2. THE LIVE RGB INFERENCE LOOP
# ==============================================================================
def main():
    weights_path = 'vein_hunter_FINAL_SKIN_weights.pth' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Booting Live Vein Hunter (RGB) on device: {device.type.upper()}")

    # --- Load Model ---
    model = VeinHunterUNet().to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print("[*] Neural Network weights loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load weights: {e}")
        return

    # --- Setup Camera ---
    cap = cv2.VideoCapture(0) # Change to 1 if it grabs the wrong camera
    if not cap.isOpened():
        print("[!] Error: Could not access the webcam.")
        return

    time.sleep(2) 
    print("[*] Camera online. Press 'q' to quit.")

    transform = transforms.Compose([transforms.ToTensor()])

    # --- Live Processing Loop ---
    while True:
        start_time = time.time()
        
        ret, frame_bgr = cap.read()
        if not ret:
            print("[!] Dropped frame.")
            break

        # Standardize Display Resolution (Optional, just makes window nice)
        height, width = frame_bgr.shape[:2]
        new_width = 800
        new_height = int((new_width / width) * height)
        display_frame = cv2.resize(frame_bgr, (new_width, new_height))
        
        # Convert to RGB for the AI
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256 exactly as done in training
        pil_img = Image.fromarray(frame_rgb).resize((256, 256))
        
        # To PyTorch Tensor
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            prediction = model(input_tensor)
            pred_prob = prediction.squeeze().cpu().numpy()
            
        # Binary mask (256x256)
        binary_mask_256 = (pred_prob > 0.5).astype(np.uint8)

        # Scale the 256x256 mask back up to the display size
        binary_mask_full = cv2.resize(binary_mask_256, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        # Create overlay
        overlay = display_frame.copy()
        # Paint veins bright green
        overlay[binary_mask_full == 1] = [0, 255, 0] 

        # Calculate & Display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the result
        cv2.imshow('Vein Hunter - Live RGB Targeting', overlay)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()