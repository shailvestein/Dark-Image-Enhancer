import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown
import os
import streamlit as st
import einops

#----------------------------------------------------------------------------------#
#                    M O D E L         A R C H I T E C T U R E                     #
#----------------------------------------------------------------------------------#


class IlluminationEstimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(IlluminationEstimator, self).__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        input_tensor = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input_tensor)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class IG_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        illu_attn = illu_fea_trans

        q, k, v, illu_attn = map(
            lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2))
        )

        v = v * illu_attn

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return out_c + out_p

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[1, 1, 1]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim, heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)

        # Encoder Loop
        fea_encoder = []
        illu_fea_list = []
        for (igab_block, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = igab_block(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder Loop
        for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlock(fea, illu_fea)

        return self.mapping(fea) + x




class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=32, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.estimator = IlluminationEstimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks)

    def forward(self, img):
        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea)
        return output_img

    def fit(self, loader, total_iterations, criterion, optimizer, scheduler, save_path, model_name, itrs_k=1000):
        print(f"Training started...\n")
        criterion = criterion.to(self.device)
        self.to(self.device)
        self.train()
        itrs = 1
        tloss = 0
        st = time.time()
        while True:
            for x, y in loader:
                optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                enh = self(x)
                enh = torch.clamp(enh, 0, 1)
                loss = criterion(enh, y)
                tloss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()
                
                if itrs%itrs_k == 0:
                    l = tloss / itrs
                    et = time.time() - st
                    print(f"Iterations: {itrs}/{total_iterations}, train_loss: {l:.6f}, time-taken: {et:.2f} sec")
                    self.save_weights(os.path.join(save_path, model_name))
                    st = time.time()
                
                if itrs>=total_iterations:
                    l = tloss / itrs
                    et = time.time() - st
                    print(f"Iterations {itrs}/{total_iterations}, train_loss: {l:.6f}, time-taken: {et:.2f} sec")
                    self.save_weights(os.path.join(save_path, model_name))
                    print("\nTraining Completed!\n")
                    return
                itrs += 1

    def predict(self, x):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            return self(x.to(self.device))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved at {path}\n")

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}\n")

    def calculate_psnr(self, low, high):
        enh = self.predict(low)
        mse = torch.mean((enh - high) ** 2)
        if mse == 0:
            return 100
        pixel_max = 1.0
        return 20 * torch.log10(pixel_max / torch.sqrt(mse))

#----------------------------------------------------------------------------------#
#            L O A D I N G    P R E - T R A I N E D   W E I G H T S                #
#----------------------------------------------------------------------------------#

def load_weights():    
    model = RetinexFormer()
    path = "./weights/"
    model_name = "retinex-multihead-transformer (12) - 100000.pth"
    model.load_weights(os.path.join(path, model_name))
    return model
