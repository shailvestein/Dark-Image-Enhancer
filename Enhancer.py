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
        Creates a mask that is 1.0 in the center and drops off sharply at edges.
        The cubic power (pow 3) ensures the center 'truth' dominates, fixing blur.
        """
        mask = torch.ones((1, patch_size, patch_size), dtype=torch.float32)
        ramp = torch.linspace(0, 1, fade_width)
        # Cubic ramp for a much sharper 'sharpness' zone
        ramp = torch.pow(ramp, 3)
        for i in range(fade_width):
            val = ramp[i]
            mask[:, i, :] *= val           # Top
            mask[:, -(i+1), :] *= val      # Bottom
            mask[:, :, i] *= val           # Left
            mask[:, :, -(i+1)] *= val      # Right
        return mask
    
    def combine_tensor_patches(self, patch_tensors, coords, original_size, padded_size, patch_size):
        h, w = original_size
        nh, nw = padded_size
        canvas = torch.zeros((3, nh, nw), dtype=torch.float32)
        weight_sum = torch.zeros((1, nh, nw), dtype=torch.float32)
        
        mask = self.get_ultra_sharp_mask(patch_size, fade_width=32)
        
        for idx, (i, j) in enumerate(coords):
            # Patch format: (C, H, W)
            patch = patch_tensors[idx].float()
            
            # Range check: ensure [0, 1]
            if patch.max() > 1.5:
                patch = patch / 255.0
                
            canvas[:, i:i+patch_size, j:j+patch_size] += (patch * mask)
            weight_sum[:, i:i+patch_size, j:j+patch_size] += mask
    
        # 1. Normalize by weights
        full_tensor = canvas / (weight_sum + 1e-8)
        
        # 2. Final Clamp to [0, 1] - CRITICAL
        full_tensor = torch.clamp(full_tensor, 0, 1)
        
        # 3. Permute to (H, W, C) for Image format
        img_np = full_tensor.permute(1, 2, 0).numpy()
    
        # 4. Safer Contrast Adjustment (Only if needed)
        p98 = np.percentile(img_np, 98)
        if 0.01 < p98 < 0.9: 
            img_np = np.clip(img_np / (p98 + 1e-6), 0, 1)
    
        # 5. Final Conversion
        final_img = (img_np * 255.0).astype(np.uint8)
        
        # Precise Crop to original dimensions
        return final_img[:h, :w, :]

    
    def enhance_image(self, img):
        start_time = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patch_size = 256 # Run inference over this patch size
        stride = 128  # Essential 50% overlap for spline blending
        h, w, _ = img.shape
        # Padding to match stride logic
        pad_h = (patch_size - h % stride) % stride + (patch_size - stride)
        pad_w = (patch_size - w % stride) % stride + (patch_size - stride)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        nh, nw, _ = img.shape
        # Extract Patches
        patches = []
        coords = []
        for i in range(0, nh - patch_size + 1, stride):
            for j in range(0, nw - patch_size + 1, stride):
                p = img[i:i+patch_size, j:j+patch_size, :]
                patches.append(torch.from_numpy(p).permute(2, 0, 1).float() / 255.0)
                coords.append((i, j))
        # Inference
        loader = DataLoader(patches, self.batch_size)
        enhanced_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch.to(self.device))
                enhanced_list.extend([p.cpu() for p in torch.clip(out, 0, 1)])
        output = self.combine_tensor_patches(enhanced_list, coords, (h, w), (nh, nw), patch_size)
        return output, time.time()-start_time 

