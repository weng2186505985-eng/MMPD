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
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

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
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

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

class PatchConsistentMLP(nn.Module):
    def __init__(self, d_model, patch_len, d_ff):
        super(PatchConsistentMLP, self).__init__()
        # Input: [current_patch_noise, hidden_H, time_emb, left_noise, right_noise]
        # But MMPD uses more consistent way. Let's simplify to a solid MLP first.
        self.mlp = nn.Sequential(
            nn.Linear(d_model + patch_len + 64 + patch_len*2, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, patch_len)
        )
        
    def forward(self, x_t, H, t_emb):
        # x_t: [B*C, N, P]
        # H: [B*C, N, D]
        # t_emb: [B*C, N, 64]
        
        B_C, N, P = x_t.shape
        
        # Consistent context: pad x_t to get neighbors
        x_padded = F.pad(x_t, (0, 0, 1, 1), mode='constant', value=0)
        left_x = x_padded[:, :-2, :] # [B*C, N, P]
        right_x = x_padded[:, 2:, :] # [B*C, N, P]
        
        feat = torch.cat([x_t, H, t_emb, left_x, right_x], dim=-1) # [B*C, N, P+D+64+2P]
        out = self.mlp(feat)
        return out

class MMPD(nn.Module):
    def __init__(self, config):
        super(MMPD, self).__init__()
        self.enc_in = config['enc_in']
        self.patch_len = config['patch_len']
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        
        self.revin = RevIN(config['enc_in'])
        self.backbone = PatchTSTBackbone(
            config['enc_in'], config['seq_len'], config['patch_len'], 
            config['patch_stride'], config['d_model'], config['n_heads'], 
            config['d_ff'], config['n_layers'], config['dropout']
        )
        self.denoiser = PatchConsistentMLP(config['d_model'], config['patch_len'], config['d_ff'])
        
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
        x_cond_norm = self.revin(x_cond, 'norm')
        H, N_cond = self.backbone(x_cond_norm) # [B*C, N_cond, D]
        
        N_target = self.pred_len // self.patch_len
        target_H = H[:, -N_target:, :] # [B*C, N_target, D]
        
        recon_patched = self.recon_head(target_H) # [B*C, N_target, P]
        
        # Denorm x_0 targets for MSE
        x_0_patched = x_0.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len)
        B, C, N_target, P = x_0_patched.shape
        x_0_patched = x_0_patched.reshape(B*C, N_target, P)
        x_0_patched_norm = (x_0_patched - self.revin.mean.reshape(-1, 1, 1)) / self.revin.stdev.reshape(-1, 1, 1)
        
        mse = F.mse_loss(recon_patched, x_0_patched_norm, reduction='none').mean(dim=(1,2))
        return mse # [B*C]

    def forward(self, x_cond, x_0, lambda_recon=0.1):
        # Training forward: DDPM + Reconstruction (Bug 4)
        B, L, C = x_cond.shape
        x_cond_norm = self.revin(x_cond, 'norm')
        H, N_cond = self.backbone(x_cond_norm)
        
        N_target = self.pred_len // self.patch_len
        target_H = H[:, -N_target:, :]
        
        # 1. Reconstruction Loss (Stage 1)
        recon_patched = self.recon_head(target_H)
        
        # Patch x_0
        x_0_patched = x_0.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len)
        x_0_patched = x_0_patched.reshape(-1, N_target, self.patch_len) # [B*C, N_target, P]
        
        # Norm with stats
        mean = self.revin.mean.permute(0, 2, 1).reshape(-1, 1, 1) 
        stdev = self.revin.stdev.permute(0, 2, 1).reshape(-1, 1, 1)
        x_0_norm = (x_0_patched - mean) / (stdev + 1e-6)
        
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
        
        return loss_diff, loss_recon

    @torch.no_grad()
    def sample(self, x_cond, num_samples=50, ddim_steps=20, gmm_K=5):
        # Bug 5: Batch Robustness
        assert x_cond.shape[0] == 1, f"sample() only supports B=1, got B={x_cond.shape[0]}"
        
        B, L, C = x_cond.shape
        x_cond_norm = self.revin(x_cond, 'norm') # Update stats
        H, N_cond = self.backbone(x_cond_norm)
        N_target = self.pred_len // self.patch_len
        target_H_single = H[:, -N_target:, :] # [C, N_target, D]
        
        # Repeat context for num_samples trajectories
        target_H = target_H_single.repeat(num_samples, 1, 1) # [num_samples * C, N_target, D]
        
        # DDIM initialization (in normalized space)
        x_t = torch.randn((num_samples * C, N_target, self.patch_len), device=x_cond.device)
        
        # Bug 1: Evolving Variational GMM Initialization
        # Use small random values for mu instead of standard randn to prevent early divergence
        pi = torch.ones(C, gmm_K, device=x_cond.device) / gmm_K
        mu = torch.randn(C, gmm_K, N_target, self.patch_len, device=x_cond.device) * 0.1
        sigma = torch.ones(C, gmm_K, device=x_cond.device)
        
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
            
            # Update GMM parameters (Vectorized EM step across all channels)
            # samples_reshaped: [num_samples, C, N_target, P]
            samples_c = x_0_est.reshape(num_samples, C, -1).transpose(0, 1) # [C, num_samples, FP]
            FP = samples_c.shape[-1]
            
            # 1. E-step: Compute responsibilities
            # diff: [C, num_samples, K, FP]
            diff = samples_c.unsqueeze(2) - mu.reshape(C, 1, gmm_K, FP)
            dist_sq = (diff**2).sum(dim=-1) # [C, num_samples, K]
            
            # log_resp: [C, num_samples, K]
            log_pi = torch.log(pi.unsqueeze(1) + 1e-6)
            log_sigma = torch.log(sigma.unsqueeze(1) + 1e-6)
            log_resp = log_pi - FP * log_sigma - 0.5 * dist_sq / (sigma.unsqueeze(1)**2 + 1e-6)
            resp = F.softmax(log_resp, dim=2) # [C, num_samples, K]
            
            # 2. M-step: Update parameters
            N_k = resp.sum(dim=1) # [C, K]
            pi = N_k / (num_samples + 1e-6)
            
            # Update mu: [C, K, FP]
            # Use batch matrix multiplication: [C, K, num_samples] @ [C, num_samples, FP]
            mu_flat = torch.bmm(resp.transpose(1, 2), samples_c) / (N_k.unsqueeze(2) + 1e-6)
            mu = mu_flat.reshape(C, gmm_K, N_target, self.patch_len)
            
            # Update sigma: [C, K]
            sigma = torch.sqrt((resp * dist_sq).sum(dim=1) / (N_k * FP + 1e-6)) + 1e-3

        # Final prediction: K modes
        # mu: [C, K, N_target, P] -> [K, pred_len, C]
        mu_out = mu.permute(1, 0, 2, 3).reshape(gmm_K, C, self.pred_len).permute(0, 2, 1)
        
        # Bug 2: Complete RevIN Denormalization
        if self.revin.affine:
            # weight/bias are [C]
            weight = self.revin.affine_weight.view(1, 1, C)
            bias = self.revin.affine_bias.view(1, 1, C)
            mu_out = (mu_out - bias) / (weight + 1e-6)
            
        mu_out = mu_out * self.revin.stdev + self.revin.mean
        
        return mu_out, pi, sigma
