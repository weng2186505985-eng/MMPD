import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.mmpd_model import PatchTSTBackbone, RevIN

class PatchTSTBaseline(nn.Module):
    """
    Pure PatchTST model for forecasting-based anomaly detection.
    """
    def __init__(self, config):
        super(PatchTSTBaseline, self).__init__()
        self.num_tm = config.get('num_tm', config['enc_in'])
        self.num_tc = config.get('num_tc', 0)
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.patch_len = config['patch_len']
        
        self.revin = RevIN(self.num_tm)
        self.backbone = PatchTSTBackbone(
            self.num_tm, config['seq_len'], config['patch_len'], 
            config['patch_stride'], config['d_model'], config['n_heads'], 
            config['d_ff'], config['n_layers'], config['dropout']
        )
        
        # Calculate number of patches
        self.num_patch = (max(self.seq_len, self.patch_len) - self.patch_len) // config['patch_stride'] + 1
        
        if self.num_tc > 0:
            self.tc_proj = nn.Linear(self.num_tc * self.patch_len, config['d_model'])
            
        # Projection from flattened patches to pred_len
        # The backbone outputs [B*num_tm, num_patch, d_model]
        # We flatten num_patch * d_model and project to pred_len
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patch * config['d_model'], self.pred_len)
        )

    def forward(self, x_cond):
        B, L, _ = x_cond.shape
        x_cond_tm = x_cond[..., :self.num_tm]
        
        x_cond_norm = self.revin(x_cond_tm, 'norm')
        H, N_cond = self.backbone(x_cond_norm) # [B*num_tm, N_cond, D]
        
        if self.num_tc > 0:
            x_cond_tc = x_cond[..., self.num_tm:]
            tc_patched = x_cond_tc.permute(0, 2, 1).unfold(-1, self.patch_len, config['patch_stride'])
            tc_patched = tc_patched.reshape(B, self.num_tc, N_cond, self.patch_len).permute(0, 2, 1, 3).reshape(B, N_cond, -1)
            tc_emb = self.tc_proj(tc_patched)
            tc_emb = tc_emb.repeat_interleave(self.num_tm, dim=0)
            H = H + tc_emb
            
        pred = self.head(H) # [B*num_tm, pred_len]
        pred = pred.reshape(B, self.num_tm, self.pred_len).permute(0, 2, 1) # [B, pred_len, num_tm]
        
        # Denorm
        if self.revin.affine:
            weight = self.revin.affine_weight.view(1, 1, self.num_tm)
            bias = self.revin.affine_bias.view(1, 1, self.num_tm)
            pred = (pred - bias) / (weight + 1e-6)
            
        pred = pred * self.revin.stdev.unsqueeze(1) + self.revin.mean.unsqueeze(1)
        return pred

class USADBaseline(nn.Module):
    """
    USAD: Unsupervised Anomaly Detection on Multivariate Time Series.
    Uses an AutoEncoder architecture with dual decoders and adversarial training.
    """
    def __init__(self, config):
        super(USADBaseline, self).__init__()
        self.in_size = config['enc_in'] * config['seq_len']
        self.latent_size = config.get('latent_size', 64)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_size, self.in_size // 2),
            nn.ReLU(True),
            nn.Linear(self.in_size // 2, self.in_size // 4),
            nn.ReLU(True),
            nn.Linear(self.in_size // 4, self.latent_size)
        )
        
        # Decoder 1
        self.decoder1 = nn.Sequential(
            nn.Linear(self.latent_size, self.in_size // 4),
            nn.ReLU(True),
            nn.Linear(self.in_size // 4, self.in_size // 2),
            nn.ReLU(True),
            nn.Linear(self.in_size // 2, self.in_size)
        )
        
        # Decoder 2
        self.decoder2 = nn.Sequential(
            nn.Linear(self.latent_size, self.in_size // 4),
            nn.ReLU(True),
            nn.Linear(self.in_size // 4, self.in_size // 2),
            nn.ReLU(True),
            nn.Linear(self.in_size // 2, self.in_size)
        )

    def forward(self, x):
        # x: [B, seq_len, C]
        B, L, C = x.shape
        x_flat = x.view(B, -1)
        
        z = self.encoder(x_flat)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        
        w1 = w1.view(B, L, C)
        w2 = w2.view(B, L, C)
        w3 = w3.view(B, L, C)
        return w1, w2, w3

class LSTMBaseline(nn.Module):
    """
    LSTM-NDT (Non-parametric Dynamic Thresholding) Baseline.
    A standard LSTM forecaster.
    """
    def __init__(self, config):
        super(LSTMBaseline, self).__init__()
        self.num_tm = config.get('num_tm', config['enc_in'])
        self.num_tc = config.get('num_tc', 0)
        self.hidden_size = config.get('d_model', 128)
        self.num_layers = config.get('n_layers', 2)
        self.pred_len = config['pred_len']
        
        total_in = self.num_tm + self.num_tc
        
        self.lstm = nn.LSTM(total_in, self.hidden_size, self.num_layers, batch_first=True, dropout=config.get('dropout', 0.1))
        self.head = nn.Linear(self.hidden_size, self.pred_len * self.num_tm)

    def forward(self, x_cond):
        # x_cond: [B, seq_len, C]
        out, (hn, cn) = self.lstm(x_cond)
        last_hidden = out[:, -1, :] # [B, hidden_size]
        
        pred = self.head(last_hidden) # [B, pred_len * num_tm]
        pred = pred.view(-1, self.pred_len, self.num_tm) # [B, pred_len, num_tm]
        
        return pred
