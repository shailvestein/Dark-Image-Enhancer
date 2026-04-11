import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown
import os
import streamlit as st


#----------------------------------------------------------------------------------#
#                    M O D E L         A R C H I T E C T U R E                     #
#----------------------------------------------------------------------------------#


class IlluminationEstimator(nn.Module):
    def __init__(self, ch=16):
        super(IlluminationEstimator, self).__init__()
        self.c1 = nn.Conv2d(4, ch, kernel_size=1)
        self.c2 = nn.Conv2d(ch, ch*2, kernel_size=5, padding=2)
        self.c3 = nn.Conv2d(ch*2, 3, kernel_size=1)

    def forward(self, x):
        x_prior = torch.max(x, dim=1, keepdim=True)[0]
        x_concat = torch.concat([x_prior, x], dim=1)
        x_feat = self.c1(x_concat)
        x_feat = nn.GELU()(x_feat)
        x_feat = self.c2(x_feat)
        x_feat = nn.GELU()(x_feat)

        x_map = self.c3(x_feat)
        x_map =torch.sigmoid(x_map)
        x_map = x_map * x

        return x_map, x_feat


class MultiHeadWindowsAttention(nn.Module):
    def __init__(self, embed_dim, xfeat_ch=32, num_head=8, window_size=8): # xfeat_ch default 32
        super(MultiHeadWindowsAttention, self).__init__()
        self.embed_dims = embed_dim
        self.num_heads = num_head
        self.head_dims = embed_dim // num_head
        self.ws = window_size

        self.cmap = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        # FIX: Yahan 128 ki jagah xfeat_ch (32) use hoga
        self.cfeat = nn.Conv2d(xfeat_ch, embed_dim, 3, padding=1)

        self.qxmap = nn.Linear(embed_dim, embed_dim)
        self.kxmap = nn.Linear(embed_dim, embed_dim)
        self.vxmap = nn.Linear(embed_dim, embed_dim)

        self.qxfeat = nn.Linear(embed_dim, embed_dim)
        self.kxfeat = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.cout = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.ws, self.ws, W // self.ws, self.ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.reshape(-1, self.ws * self.ws, C)
        return windows, H, W

    def window_reverse(self, windows, H, W):
        nW = (H // self.ws) * (W // self.ws)
        B = windows.shape[0] // nW
        x = windows.view(B, H // self.ws, W // self.ws, self.ws, self.ws, self.embed_dims)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(B, self.embed_dims, H, W)

    def forward(self, xmap, xfeat):
        B, C, H, W = xmap.shape
        if xfeat.shape[2:] != xmap.shape[2:]:
            xfeat = F.interpolate(xfeat, size=(H, W), mode='bilinear', align_corners=False)

        xmap = self.cmap(xmap)
        identity = xmap

        # Window Partitioning
        win_xmap, _, _ = self.window_partition(xmap)
        win_xfeat, _, _ = self.window_partition(self.cfeat(xfeat)) # Ab 32 -> 128 safely convert hoga

        N = self.ws * self.ws
        B_win = win_xmap.shape[0]

        # Q, K, V calculations
        q_m = self.qxmap(win_xmap).view(B_win, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        k_m = self.kxmap(win_xmap).view(B_win, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        v_m = self.vxmap(win_xmap).view(B_win, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        q_f = self.qxfeat(win_xfeat).view(B_win, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        k_f = self.kxfeat(win_xfeat).view(B_win, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        # Attention Score
        q_guided = torch.sigmoid(q_f) * q_m
        k_guided = torch.sigmoid(k_f) * k_m
        attn = (q_guided @ k_guided.transpose(-2, -1)) / (self.head_dims ** 0.5)
        attn = attn.softmax(dim=-1)

        # Output
        out = (attn @ v_m).permute(0, 2, 1, 3).contiguous().view(B_win, N, self.embed_dims)
        out = self.window_reverse(self.proj(out), H, W)
        return self.cout(out) + identity
    

class IGAB(nn.Module):
    def __init__(self, in_ch, embed_dim, xfeat_ch=32):
        super(IGAB, self).__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, 1) if in_ch != embed_dim else nn.Identity()

        self.norm1 = nn.LayerNorm(embed_dim)
        # Yahan hum xfeat_ch pass kar rahe hain
        self.attn = MultiHeadWindowsAttention(embed_dim=embed_dim, xfeat_ch=xfeat_ch)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim*2, embed_dim*2, 3, padding=1, groups=embed_dim*2),
            nn.GELU(),
            nn.Conv2d(embed_dim*2, embed_dim, 1)
        )

    def forward(self, xmap, xfeat):
        x = self.proj(xmap)
        identity = x

        # Norm + Attention
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x).permute(0, 3, 1, 2)
        x = self.attn(x, xfeat) + identity

        # Norm + FFN
        identity = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x).permute(0, 3, 1, 2)
        x = self.ffn(x) + identity

        return x
    

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.down(x)



class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.up(x)
    

class RetinexFormer(nn.Module):
    def __init__(self, embed_dims=[64, 128, 256]):
        super(RetinexFormer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Removed self.criterion, self.optimizer, self.scheduler from here

        self.ie = IlluminationEstimator()

        self.enc_proj = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )

        self.enc_block1 = IGAB(in_ch=embed_dims[0], embed_dim=embed_dims[0])
        self.down1 = DownSample(embed_dims[0], embed_dims[1])

        self.enc_block2 = IGAB(in_ch=embed_dims[1], embed_dim=embed_dims[1])
        self.down2 = DownSample(embed_dims[1], embed_dims[2])

        self.bottleneck = IGAB(in_ch=embed_dims[2], embed_dim=embed_dims[2])

        self.up1 = UpSample(embed_dims[2], embed_dims[1])
        self.dec_block1 = IGAB(in_ch=embed_dims[1], embed_dim=embed_dims[1])

        self.up2 = UpSample(embed_dims[1], embed_dims[0])
        self.dec_block2 = IGAB(in_ch=embed_dims[0], embed_dim=embed_dims[0])

        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dims[0], 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.GELU()
        )

    def forward(self, x):
        xmap, xfeat = self.ie(x)

        e1_input = self.enc_proj(xmap)
        e1 = self.enc_block1(e1_input, xfeat) # level-1 (128, 128, 64)

        xfeat_d1 = F.interpolate(xfeat, scale_factor=0.5, mode='bilinear', align_corners=False)
        e2_input = self.down1(e1)
        e2 = self.enc_block2(e2_input, xfeat_d1) # level-2 (64, 64, 128)

        xfeat_d2 = F.interpolate(xfeat, scale_factor=0.25, mode='bilinear', align_corners=False)
        bottleneck_input = self.down2(e2)
        bottleneck = self.bottleneck(bottleneck_input, xfeat_d2) #level-3 (32, 32, 256)

        d1_input = self.up1(bottleneck)
        d1 = d1_input + e2              # residual skip
        d1 = self.dec_block1(d1, xfeat_d1) # decoder level-2 (64, 64, 128)

        d2_input = self.up2(d1)
        d2 = d2_input + e1              # residual skip
        d2 = self.dec_block2(d2, xfeat) # decoder level-1 (128, 128, 64)

        out = self.final_conv(d2)
        return out + x # Global residual learning (Input image + learned correction)


    def fit(self, train_loader, val_loader, epochs, criterion, optimizer, scheduler, save_path, model_name):
        criterion = criterion.to(self.device)
        self.to(self.device)
        epoch = 1
        iterations = 0
        best_psnr = 0
        while epoch <= epochs:
            train_loss_, val_loss_, psnr = 0, 0, 0
            for x, y in tqdm(train_loader):
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                enh = self(x)
                enh = torch.clamp(enh, 0, 1)
                train_loss = criterion(enh, y)
                train_loss_ += train_loss.item()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()
                iterations += 1

                if iterations%1000 == 0:
                    vi = 0
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        with torch.no_grad():
                            enh = self(x)
                        val_loss = criterion(enh, y)
                        val_loss_ += val_loss.item()
                        psnr += self.calculate_psnr(enh, y)
                        if vi == 10:
                            break
                        vi += 1
                    psnr = psnr/vi
                    print(f"Epoch: {epoch}/{epochs}, train_loss: {train_loss_/len(train_loader):.4f}, val_loss: {val_loss_/vi:.4f}, PSNR: {psnr} lr_rate: {optimizer.param_groups[0]['lr']}")
            
                    if psnr>best_psnr:
                        best_psnr = psnr
                        print(f"End of epoch: {epoch}/{epochs} saving model with best PSNR: {best_psnr}\t", end='')
                        self.save_weights(os.path.join(save_path, model_name))
            epoch += 1
            

    def predict(self, x):
        self.to(self.device)
        with torch.no_grad():
            return self(x.to(self.device))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved at {path}\n")
        return

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}\n")
        return


    def calculate_psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        pixel_max = 1.0
        return 20 * torch.log10(pixel_max / torch.sqrt(mse))




#----------------------------------------------------------------------------------#
#            D O W N L O A D    P R E - T R A I N E D   W E I G H T S              #
#----------------------------------------------------------------------------------#

model_id = "1u9GPI0n-Wys9dkzIPGbusJW_P04_BTob"
model_name = "retinex-multihead-transformer.pth"


def download_weights(file_id, model_name):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_name):
        with st.spinner(f"Downloading model {model_name} weights from Google Drive..."):
            gdown.download(url, model_name, quiet=False)
    return model_name

#----------------------------------------------------------------------------------#
#            L O A D I N G    P R E - T R A I N E D   W E I G H T S                #
#----------------------------------------------------------------------------------#

def load_weights():    
    download_weights(model_id, model_name)
    model = RetinexFormer()
    model.load_weights(model_name)
    return model
