import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        x_fp32 = x.float()
        self.mean = torch.mean(x_fp32, dim=dim2reduce, keepdim=True).to(x.dtype).detach()
        self.stdev = torch.sqrt(torch.var(x_fp32, dim=dim2reduce, keepdim=True, unbiased=False) + 1e-4).to(x.dtype).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight
        x = x * self.stdev
        x = x + self.mean
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PatchTSTBackbone(nn.Module):
    def __init__(self, enc_in, seq_len, patch_len, stride, d_model, n_heads, d_ff, n_layers, dropout):
        super(PatchTSTBackbone, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
        
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
    def forward(self, x):
        # x: [Batch, Seq, Channel]
        B, L, C = x.shape
        x = x.permute(0, 2, 1) # [B, C, L]
        
        # Patching
        # Pad maybe if needed, but let's assume L is compatible
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B, C, N, P]
        N = x.shape[2]
        
        x = x.reshape(-1, N, self.patch_len) # [B*C, N, P]
        
        # Embedding
        x = self.value_embedding(x) # [B*C, N, D]
        x = self.pos_encoding(x)
        
        # Transformer
        enc_out = self.encoder(x) # [B*C, N, D]
        
        return enc_out, N

class TransformerDenoiser(nn.Module):
    def __init__(self, d_model, patch_len, d_ff, n_heads=4, n_layers=2):
        super(TransformerDenoiser, self).__init__()
        self.input_proj = nn.Linear(patch_len + d_model + 64, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, patch_len)
        
    def forward(self, x_t, H, t_emb):
        # x_t: [B*C, N, P]
        # H: [B*C, N, D]
        # t_emb: [B*C, N, 64]
        feat = torch.cat([x_t, H, t_emb], dim=-1) # [B*C, N, P+D+64]
        x = self.input_proj(feat) # [B*C, N, D]
        out = self.transformer(x) # [B*C, N, D]
        return self.out_proj(out)

class PatchConsistentMLP(nn.Module):
    def __init__(self, d_model, patch_len, d_ff):
        super(PatchConsistentMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + patch_len + 64 + patch_len*2, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, patch_len)
        )
        
    def forward(self, x_t, H, t_emb):
        B_C, N, P = x_t.shape
        x_padded = F.pad(x_t, (0, 0, 1, 1), mode='constant', value=0)
        left_x = x_padded[:, :-2, :]
        right_x = x_padded[:, 2:, :]
        feat = torch.cat([x_t, H, t_emb, left_x, right_x], dim=-1)
        return self.mlp(feat)

class ChannelMixing(nn.Module):
    def __init__(self, num_tm):
        super(ChannelMixing, self).__init__()
        self.num_tm = num_tm
        self.mixing = nn.Linear(num_tm, num_tm)
        
    def forward(self, H):
        # H: [B*num_tm, N, D]
        B_num_tm, N, D = H.shape
        B = B_num_tm // self.num_tm
        H = H.view(B, self.num_tm, N, D).permute(0, 2, 3, 1) # [B, N, D, num_tm]
        H = self.mixing(H) # [B, N, D, num_tm]
        H = H.permute(0, 3, 1, 2).reshape(B_num_tm, N, D)
        return H

class MMPD(nn.Module):
    def __init__(self, config):
        super(MMPD, self).__init__()
        self.enc_in = config['enc_in']
        self.patch_len = config['patch_len']
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        
        self.num_tm = config.get('num_tm', config['enc_in'])
        self.num_tc = config.get('num_tc', 0) if not config.get('ablation_no_tc', False) else 0
        self.patch_stride = config['patch_stride']
        self.ablation_no_uw = config.get('ablation_no_uw', False)
        
        self.revin = RevIN(self.num_tm)
        self.backbone = PatchTSTBackbone(
            self.num_tm, config['seq_len'], config['patch_len'], 
            config['patch_stride'], config['d_model'], config['n_heads'], 
            config['d_ff'], config['n_layers'], config['dropout']
        )
        if self.num_tc > 0:
            self.tc_proj = nn.Linear(self.num_tc * self.patch_len, config['d_model'])
            
        self.channel_mixing = ChannelMixing(self.num_tm)
            
        if config.get('ablation_no_transformer', False):
            self.denoiser = PatchConsistentMLP(config['d_model'], config['patch_len'], config['d_ff'])
        else:
            self.denoiser = TransformerDenoiser(config['d_model'], config['patch_len'], config['d_ff'], config['n_heads'])
        
        # Time embedding
        self.t_embedder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Diffusion parameters
        num_steps = config.get('num_diff_steps', 100)
        self.register_buffer('beta', self._cosine_beta_schedule(num_steps))
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        self.num_steps = num_steps
        
        # Stage 1 reconstruction head
        self.recon_head = nn.Linear(config['d_model'], config['patch_len'])
        
        # Uncertainty Weighting for Multi-task learning
        self.log_var_diff = nn.Parameter(torch.zeros(1))
        self.log_var_recon = nn.Parameter(torch.zeros(1))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_bar = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def get_mse_recon(self, x_cond, x_0):
        # Stage 1: Rough screening using backbone H to predict x_0
        B, L, C = x_cond.shape
        x_cond_tm = x_cond[..., :self.num_tm]
        
        x_cond_norm = self.revin(x_cond_tm, 'norm')
        H, N_cond = self.backbone(x_cond_norm) # [B*num_tm, N_cond, D]
        
        if self.num_tc > 0:
            x_cond_tc = x_cond[..., self.num_tm:] # [B, L, num_tc]
            tc_patched = x_cond_tc.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_stride)
            tc_patched = tc_patched.reshape(B, self.num_tc, N_cond, self.patch_len).permute(0, 2, 1, 3).reshape(B, N_cond, -1)
            tc_emb = self.tc_proj(tc_patched) # [B, N_cond, D]
            tc_emb = tc_emb.repeat_interleave(self.num_tm, dim=0) # [B*num_tm, N_cond, D]
            H = H + tc_emb
        
        N_target = self.pred_len // self.patch_len
        target_H = H[:, -N_target:, :] # [B*C, N_target, D]
        
        recon_patched = self.recon_head(target_H) # [B*C, N_target, P]
        
        # Denorm x_0 targets for MSE (only TM)
        x_0_tm = x_0[..., :self.num_tm]
        x_0_patched = x_0_tm.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len)
        B, C, N_target, P = x_0_patched.shape
        x_0_patched = x_0_patched.reshape(B*C, N_target, P)
        x_0_patched_norm = (x_0_patched - self.revin.mean.reshape(-1, 1, 1)) / self.revin.stdev.reshape(-1, 1, 1)
        
        mse = F.mse_loss(recon_patched, x_0_patched_norm, reduction='none').mean(dim=(1,2))
        return mse # [B*C]

    def forward(self, x_cond, x_0, lambda_recon=0.1):
        # Training forward: DDPM + Reconstruction
        B, L, C = x_cond.shape
        x_cond_tm = x_cond[..., :self.num_tm]
        x_cond_norm = self.revin(x_cond_tm, 'norm')
        H, N_cond = self.backbone(x_cond_norm)
        
        if self.num_tc > 0:
            x_cond_tc = x_cond[..., self.num_tm:]
            tc_patched = x_cond_tc.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_stride)
            tc_patched = tc_patched.reshape(B, self.num_tc, N_cond, self.patch_len).permute(0, 2, 1, 3).reshape(B, N_cond, -1)
            tc_emb = self.tc_proj(tc_patched)
            tc_emb = tc_emb.repeat_interleave(self.num_tm, dim=0)
            H = H + tc_emb
            
        H = self.channel_mixing(H)
        
        N_target = self.pred_len // self.patch_len
        target_H = H[:, -N_target:, :]
        
        # 1. Reconstruction Loss (Stage 1)
        recon_patched = self.recon_head(target_H)
        
        # Patch x_0 (only TM)
        x_0_tm = x_0[..., :self.num_tm]
        x_0_patched = x_0_tm.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len)
        x_0_patched = x_0_patched.reshape(-1, N_target, self.patch_len) # [B*C, N_target, P]
        
        # Norm with stats
        mean = self.revin.mean.permute(0, 2, 1).reshape(-1, 1, 1) 
        stdev = self.revin.stdev.permute(0, 2, 1).reshape(-1, 1, 1)
        x_0_norm = (x_0_patched - mean) / (stdev + 1e-4)
        
        # Apply affine if needed
        if self.revin.affine:
            w = self.revin.affine_weight.repeat(B).reshape(-1, 1, 1)
            b = self.revin.affine_bias.repeat(B).reshape(-1, 1, 1)
            x_0_norm = x_0_norm * w + b
        
        loss_recon = F.mse_loss(recon_patched, x_0_norm)
        
        # 2. Diffusion Loss (DDPM)
        # Sample t
        t = torch.randint(0, self.num_steps, (x_0_norm.shape[0],), device=x_cond.device)
        noise = torch.randn_like(x_0_norm)
        
        # Add noise
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0_norm + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise
        t_float = t.float().view(-1, 1, 1) / self.num_steps
        t_emb = self.t_embedder(t_float) # [B*C, 1, 64]
        t_emb = t_emb.expand(-1, N_target, -1)
        noise_pred = self.denoiser(x_t, target_H, t_emb)
        
        loss_diff = F.mse_loss(noise_pred, noise)
        
        if self.ablation_no_uw:
            total_loss = loss_diff + lambda_recon * loss_recon
        else:
            loss_diff_w = torch.exp(-self.log_var_diff) * loss_diff + self.log_var_diff
            loss_recon_w = torch.exp(-self.log_var_recon) * loss_recon + self.log_var_recon
            total_loss = loss_diff_w + loss_recon_w
        
        return total_loss, loss_diff, loss_recon

    @torch.no_grad()
    def sample(self, x_cond, num_samples=50, ddim_steps=20, gmm_K=5):
        B, L, C_all = x_cond.shape
        x_cond_tm = x_cond[..., :self.num_tm]
        x_cond_norm = self.revin(x_cond_tm, 'norm') # Update stats
        H, N_cond = self.backbone(x_cond_norm)
        
        if self.num_tc > 0:
            x_cond_tc = x_cond[..., self.num_tm:]
            tc_patched = x_cond_tc.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_stride)
            tc_patched = tc_patched.reshape(B, self.num_tc, N_cond, self.patch_len).permute(0, 2, 1, 3).reshape(B, N_cond, -1)
            tc_emb = self.tc_proj(tc_patched)
            tc_emb = tc_emb.repeat_interleave(self.num_tm, dim=0)
            H = H + tc_emb
            
        H = self.channel_mixing(H)
            
        N_target = self.pred_len // self.patch_len
        target_H_single = H[:, -N_target:, :] # [B*num_tm, N_target, D]
        
        # Recon Initial Prior for GMM Mean
        recon_patched = self.recon_head(target_H_single) # [B*num_tm, N_target, P]
        recon_patched = recon_patched.reshape(B, self.num_tm, 1, N_target, self.patch_len) # [B, num_tm, 1, N_target, P]
        
        # Repeat context for num_samples trajectories
        target_H = target_H_single.repeat(num_samples, 1, 1) # [num_samples * B * num_tm, N_target, D]
        
        # DDIM initialization (in normalized space)
        x_t = torch.randn((num_samples * B * self.num_tm, N_target, self.patch_len), device=x_cond.device)
        
        # Evolving Variational GMM Initialization using Recon Prior
        pi = torch.ones(B, self.num_tm, gmm_K, device=x_cond.device) / gmm_K
        mu = recon_patched.repeat(1, 1, gmm_K, 1, 1) + torch.randn(B, self.num_tm, gmm_K, N_target, self.patch_len, device=x_cond.device) * 0.05
        sigma = torch.ones(B, self.num_tm, gmm_K, device=x_cond.device)
        
        # Time steps
        times = torch.linspace(self.num_steps - 1, 0, ddim_steps + 1).long()
        
        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i+1]
            
            t_tensor = torch.full((x_t.shape[0], N_target, 1), t, device=x_cond.device).float() / self.num_steps
            t_emb = self.t_embedder(t_tensor)
            
            noise_pred = self.denoiser(x_t, target_H, t_emb)
            
            # DDIM update to get x_next and estimate x_0_est
            alpha_t = self.alpha_bar[t]
            alpha_t_next = self.alpha_bar[t_next]
            
            x_0_est = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x_t = torch.sqrt(alpha_t_next) * x_0_est + torch.sqrt(1 - alpha_t_next) * noise_pred
            
            # Update GMM parameters (Vectorized EM step across batches and channels)
            # x_0_est: [num_samples * B * num_tm, N_target, P]
            samples_c = x_0_est.reshape(num_samples, B, self.num_tm, -1).permute(1, 2, 0, 3) # [B, num_tm, num_samples, FP]
            FP = samples_c.shape[-1]
            
            # 1. E-step: Compute responsibilities
            # diff: [B, num_tm, num_samples, K, FP]
            diff = samples_c.unsqueeze(3) - mu.reshape(B, self.num_tm, 1, gmm_K, FP)
            dist_sq = (diff**2).sum(dim=-1) # [B, num_tm, num_samples, K]
            
            log_pi = torch.log(pi.unsqueeze(2) + 1e-6)
            log_sigma = torch.log(sigma.unsqueeze(2) + 1e-6)
            log_resp = log_pi - FP * log_sigma - 0.5 * dist_sq / (sigma.unsqueeze(2)**2 + 1e-6)
            resp = F.softmax(log_resp, dim=3) # [B, num_tm, num_samples, K]
            
            # 2. M-step: Update parameters
            N_k = resp.sum(dim=2) # [B, num_tm, K]
            pi = N_k / (num_samples + 1e-6)
            
            # Update mu
            # resp: [B, num_tm, num_samples, K] -> [B, num_tm, K, num_samples]
            # samples_c: [B, num_tm, num_samples, FP]
            mu_flat = torch.matmul(resp.transpose(2, 3), samples_c) / (N_k.unsqueeze(-1) + 1e-6)
            mu = mu_flat.reshape(B, self.num_tm, gmm_K, N_target, self.patch_len)
            
            # Update sigma
            sigma = torch.sqrt((resp * dist_sq).sum(dim=2) / (N_k * FP + 1e-6)) + 1e-3

        # Final prediction: K modes
        # mu: [B, num_tm, K, N_target, P] -> [B, num_tm, K, pred_len]
        mu_out = mu.permute(0, 2, 1, 3, 4).reshape(B, gmm_K, self.num_tm, self.pred_len).permute(0, 1, 3, 2) # [B, K, pred_len, num_tm]
        
        # Complete RevIN Denormalization
        if self.revin.affine:
            weight = self.revin.affine_weight.view(1, 1, 1, self.num_tm)
            bias = self.revin.affine_bias.view(1, 1, 1, self.num_tm)
            mu_out = (mu_out - bias) / (weight + 1e-4)
            
        mu_out = mu_out * self.revin.stdev.unsqueeze(1) + self.revin.mean.unsqueeze(1)
        
        return mu_out, pi, sigma
