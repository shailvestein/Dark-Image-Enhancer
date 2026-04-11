import torch
import numpy as np
import cv2
import time
from torch.utils.data import DataLoader

class Enhancer:
    def __init__(self, model, batch_size):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.batch_size = batch_size
    
    def get_ultra_sharp_mask(self, patch_size, fade_width=32):
        """
        Creates a 2D Cosine window mask for seamless blending.
        This eliminates the grid lines/seams.
        """
        mask1d = torch.ones(patch_size)
        # Cosine ramp transition
        ramp = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, fade_width)))
        mask1d[:fade_width] = ramp
        mask1d[-fade_width:] = torch.flip(ramp, [0])
        
        # Convert to 2D
        mask2d = mask1d.view(1, -1) * mask1d.view(-1, 1)
        return mask2d.unsqueeze(0) # Shape: (1, H, W)
    
    def combine_tensor_patches(self, patch_tensors, coords, original_size, padded_size, patch_size):
        h, w = original_size
        nh, nw = padded_size
        canvas = torch.zeros((3, nh, nw), dtype=torch.float32)
        weight_sum = torch.zeros((1, nh, nw), dtype=torch.float32)
        
        mask = self.get_ultra_sharp_mask(patch_size, fade_width=48) # Increased fade for better blending
        
        for idx, (i, j) in enumerate(coords):
            patch = patch_tensors[idx].float()
            if patch.max() > 1.5: patch = patch / 255.0
                
            canvas[:, i:i+patch_size, j:j+patch_size] += (patch * mask)
            weight_sum[:, i:i+patch_size, j:j+patch_size] += mask
    
        full_tensor = torch.clamp(canvas / (weight_sum + 1e-8), 0, 1)
        img_np = full_tensor.permute(1, 2, 0).numpy()
        
        # Final Conversion
        final_img = (img_np * 255.0).astype(np.uint8)
        return final_img[:h, :w, :]

    def enhance_image(self, img):
        start_time = time.time()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patch_size = 256
        stride = 192  
        h, w, _ = img_rgb.shape
        
        pad_h = (patch_size - h % stride) % stride + (patch_size - stride)
        pad_w = (patch_size - w % stride) % stride + (patch_size - stride)
        img_padded = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        nh, nw, _ = img_padded.shape
        
        patches = []
        coords = []
        for i in range(0, nh - patch_size + 1, stride):
            for j in range(0, nw - patch_size + 1, stride):
                p = img_padded[i:i+patch_size, j:j+patch_size, :]
                patches.append(torch.from_numpy(p).permute(2, 0, 1).float() / 255.0)
                coords.append((i, j))

        loader = DataLoader(patches, self.batch_size)
        enhanced_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch.to(self.device))
                enhanced_list.extend([p.cpu() for p in out])
                
        output = self.combine_tensor_patches(enhanced_list, coords, (h, w), (nh, nw), patch_size)
        return output, time.time() - start_time
