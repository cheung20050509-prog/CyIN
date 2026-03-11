"""
CyIN: Cyclic Informative Latent Space
严格按照论文公式实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IBEncoder(nn.Module):
    """
    Information Bottleneck Encoder (Eq.5-6)
    F_u → (μ, logvar) → B via reparameterization
    """
    def __init__(self, input_dim=256, hidden_dim=256, bottleneck_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim * 2)  # μ and logvar
        )
        self.bottleneck_dim = bottleneck_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
        Returns:
            B: bottleneck latent
            mu, logvar: for KL computation
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        B = self.reparameterize(mu, logvar)
        return B, mu, logvar


class IBDecoder(nn.Module):
    """
    Information Bottleneck Decoder (Eq.7-8)
    B → F_reconstructed
    """
    def __init__(self, bottleneck_dim=128, hidden_dim=256, output_dim=256, dropout=0.3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, B):
        return self.decoder(B)


class ResidualAutoencoder(nn.Module):
    """Single Residual Autoencoder block (Eq.14): RA(x) = Dec(Enc(x)) + x"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        h = F.relu(self.encoder(x))
        return self.decoder(h) + x


class CRA(nn.Module):
    """
    Cascaded Residual Autoencoder (Eq.13-14).
    Following the original CRA design (MMIN, AAAI 2021): sequential chaining of RA blocks.
    Each RA_i receives the output of RA_{i-1}, i.e. x_{i} = RA_i(x_{i-1}).
    RA(x) = Dec(Enc(x)) + x provides stability via internal residual skip.
    """
    def __init__(self, bottleneck_dim=128, num_layers=8, hidden_dims=[64, 32, 16]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            hidden_dim = hidden_dims[i % len(hidden_dims)]
            self.layers.append(ResidualAutoencoder(bottleneck_dim, hidden_dim))

    def forward(self, B):
        x = B
        for layer in self.layers:
            x = layer(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, expansion=4):
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal Attention (Eq.21)
    M_uv = LN(softmax(Q_u K_v^T / √d) V_v + Q_u)
    """
    def __init__(self, dim=128, num_heads=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.feed_forwards = nn.ModuleList([
            FeedForwardBlock(dim, dropout=dropout) for _ in range(num_layers)
        ])
    
    def forward(self, query, key_value):
        """
        Args:
            query: (batch, dim) - 查询模态
            key_value: (batch, dim) - 被注意的模态
        Returns:
            (batch, dim) - 交叉注意力输出
        """
        x = query.unsqueeze(1)  # (B, 1, D)
        kv = key_value.unsqueeze(1)  # (B, 1, D)
        
        for attn, attn_ln, ffn_ln, feed_forward in zip(
            self.layers,
            self.attn_norms,
            self.ffn_norms,
            self.feed_forwards,
        ):
            attn_query = attn_ln(x)
            attn_key_value = attn_ln(kv)
            attn_out, _ = attn(attn_query, attn_key_value, attn_key_value)
            z = attn_out + attn_query
            z_norm = ffn_ln(z)
            x = feed_forward(z_norm) + z_norm
        
        return x.squeeze(1)


class MultimodalFusion(nn.Module):
    """
    Multimodal Fusion (Eq.22)
    F_M = Concat([M_ta, M_tv, M_av])
    """
    def __init__(self, dim=128, num_heads=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.attn_ta = CrossModalAttention(dim, num_heads, num_layers, dropout)
        self.attn_tv = CrossModalAttention(dim, num_heads, num_layers, dropout)
        self.attn_av = CrossModalAttention(dim, num_heads, num_layers, dropout)
    
    def forward(self, B_t, B_a, B_v):
        """
        Args:
            B_t, B_a, B_v: (batch, dim) bottleneck latents
        Returns:
            F_M: (batch, 3*dim) fused representation
        """
        M_ta = self.attn_ta(B_t, B_a)  # text attends to acoustic
        M_tv = self.attn_tv(B_t, B_v)  # text attends to visual
        M_av = self.attn_av(B_a, B_v)  # acoustic attends to visual
        return torch.cat([M_ta, M_tv, M_av], dim=-1)


class CyIN(nn.Module):
    """
    CyIN: Cyclic Informative Latent Space
    Complete implementation following paper equations
    """
    def __init__(self, args):
        super().__init__()
        self.modalities = ('t', 'a', 'v')
        self.modality_to_idx = {modality: idx for idx, modality in enumerate(self.modalities)}
        
        # 从args获取参数
        text_dim = args.get('text_dim', 768)
        acoustic_dim = args.get('acoustic_dim', 74)
        visual_dim = args.get('visual_dim', 47)
        unified_dim = args.get('unified_dim', 256)       # C_U
        ib_dim = args.get('ib_dim', 256)                 # C_ib
        bottleneck_dim = args.get('bottleneck_dim', 128) # C_B
        dropout = args.get('dropout_prob', 0.3)
        
        # CRA参数 (Table 4: asymmetric layer allocation la:lv:av)
        cra_layers = args.get('cra_layers', 8)
        cra_dims = args.get('cra_dims', [64, 32, 16])
        cra_layers_la = args.get('cra_layers_la') or cra_layers
        cra_layers_lv = args.get('cra_layers_lv') or cra_layers
        cra_layers_av = args.get('cra_layers_av') or cra_layers
        
        # Attention参数
        attn_layers = args.get('attention_layers', 2)
        attn_heads = args.get('attention_heads', 2)
        
        # IB参数
        self.beta = args.get('beta', 32)
        self.gamma = args.get('gamma', 10)
        self.normalize_av = args.get('normalize_av', False)
        
        # ============ 1. Unimodal Projectors ============
        self.proj_t = nn.Sequential(
            nn.Linear(text_dim, unified_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(acoustic_dim, unified_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(visual_dim, unified_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============ 2. IB Encoders ============
        # 每个模态独立的IB编码器
        self.ib_enc_t = IBEncoder(unified_dim, ib_dim, bottleneck_dim, dropout)
        self.ib_enc_a = IBEncoder(unified_dim, ib_dim, bottleneck_dim, dropout)
        self.ib_enc_v = IBEncoder(unified_dim, ib_dim, bottleneck_dim, dropout)
        
        # ============ 3. IB Decoders ============
        # 9个解码器: 3 intra-modal (S→S) + 6 inter-modal (S→T)
        self.decoders = nn.ModuleDict({
            # Intra-modal: 自重构
            't_t': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
            'a_a': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
            'v_v': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
            # Inter-modal: 互重构
            't_a': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
            't_v': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
            'a_t': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
            'a_v': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
            'v_t': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
            'v_a': IBDecoder(bottleneck_dim, ib_dim, unified_dim, dropout),
        })
        
        # ============ 4. CRA Translators ============
        # Paper Table 4: asymmetric layer allocation (la:lv:av = 4:4:1 best)
        self.translators = nn.ModuleDict({
            'a_t': CRA(bottleneck_dim, cra_layers_la, cra_dims),
            't_v': CRA(bottleneck_dim, cra_layers_lv, cra_dims),
            'a_v': CRA(bottleneck_dim, cra_layers_av, cra_dims),
        })
        
        # ============ 5. Multimodal Fusion ============
        self.fusion = MultimodalFusion(bottleneck_dim, attn_heads, attn_layers, dropout)
        
        # ============ 6. Label Predictors (for L_lib) ============
        self.label_preds = nn.ModuleDict({
            't': nn.Linear(bottleneck_dim, 1),
            'a': nn.Linear(bottleneck_dim, 1),
            'v': nn.Linear(bottleneck_dim, 1),
        })
        
        # ============ 7. Final Predictor ============
        self.predictor = nn.Sequential(
            nn.Linear(bottleneck_dim * 3, unified_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(unified_dim, 1)
        )

    def _prepare_modality_mask(self, modality_mask, batch_size, device, dtype):
        if modality_mask is None:
            return torch.ones(batch_size, len(self.modalities), device=device, dtype=dtype)
        if modality_mask.dim() == 1:
            modality_mask = modality_mask.unsqueeze(0)
        return modality_mask.to(device=device, dtype=dtype)

    def _translate_latent(self, source, target, bottleneck):
        translator_key = '_'.join(sorted((source, target)))
        translator = self.translators[translator_key]
        return translator(bottleneck)
    
    def _compute_kl_loss(self, mu, logvar, mask=None):
        """KL(q(B|X) || p(B)) where p(B) = N(0, I)."""
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if mask is None or mu.dim() == 2:
            return kl.mean()

        mask = mask.unsqueeze(-1).type_as(kl)
        return (kl * mask).sum() / (mask.sum().clamp_min(1.0) * mu.size(-1))

    def _prepare_labels(self, labels):
        """Flatten regression labels to shape (batch,) to avoid broadcast errors."""
        if labels is None:
            return None
        if labels.dim() == 0:
            return labels.unsqueeze(0)
        return labels.reshape(labels.size(0), -1).squeeze(-1)

    def _masked_mean(self, tensor, mask=None):
        """Average token sequences over valid positions only."""
        if tensor.dim() == 2:
            return tensor
        if mask is None:
            return tensor.mean(dim=1)

        mask = mask.unsqueeze(-1).type_as(tensor)
        return (tensor * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _compute_reconstruction_loss(self, reconstructed, target, mask=None):
        if mask is None or reconstructed.dim() == 2:
            return F.mse_loss(reconstructed, target)

        diff = (reconstructed - target).pow(2)
        mask = mask.unsqueeze(-1).type_as(diff)
        return (diff * mask).sum() / (mask.sum().clamp_min(1.0) * reconstructed.size(-1))
    
    def _compute_token_ib_loss(self, B_s, mu_s, logvar_s, F_t, decoder, mask=None):
        """
        Token-level IB loss (Eq.8)
        L_tib^{S→T} = KL + β * reconstruction_loss
        """
        kl_loss = self._compute_kl_loss(mu_s, logvar_s, mask)
        F_t_rec = decoder(B_s)
        rec_loss = self._compute_reconstruction_loss(F_t_rec, F_t, mask)
        return kl_loss + self.beta * rec_loss
    
    def _compute_cyclic_tib(self, F, B, mu, logvar, mask=None):
        """
        Cyclic Token-level IB (Eq.9)
        L_tib = E[L_tib^{S→S} + 0.5*(L_tib^{S→T} + L_tib^{T→S})]
        """
        L_tib = 0
        
        # Intra-modal: S→S
        for m in ['t', 'a', 'v']:
            L_tib += self._compute_token_ib_loss(
                B[m], mu[m], logvar[m], F[m], self.decoders[f'{m}_{m}'], mask
            )
        
        # Inter-modal: S→T 和 T→S
        for s, t in [('t', 'a'), ('t', 'v'), ('a', 'v')]:
            L_st = self._compute_token_ib_loss(B[s], mu[s], logvar[s], F[t], self.decoders[f'{s}_{t}'], mask)
            L_ts = self._compute_token_ib_loss(B[t], mu[t], logvar[t], F[s], self.decoders[f'{t}_{s}'], mask)
            L_tib += 0.5 * (L_st + L_ts)
        
        return L_tib
    
    def _compute_label_ib_loss(self, B, mu, logvar, labels):
        """
        Label-level IB loss (Eq.10-12 for regression).
        Compute the supervision per modality and average over modalities.
        """
        labels = self._prepare_labels(labels)
        total_loss = labels.new_tensor(0.0)
        modality_losses = {}

        for modality in ['t', 'a', 'v']:
            kl_loss = self._compute_kl_loss(mu[modality], logvar[modality])
            y_pred = self.label_preds[modality](B[modality]).squeeze(-1)
            pred_loss = F.l1_loss(y_pred, labels)
            modality_loss = kl_loss + self.beta * pred_loss
            total_loss = total_loss + modality_loss
            modality_losses[f'L_lib_{modality}'] = modality_loss.item()

        return total_loss / len(B), modality_losses

    def _compute_directional_translation_loss(self, source, target, bottlenecks):
        target_reconstructed = self._translate_latent(source, target, bottlenecks[source])
        source_cyclic = self._translate_latent(target, source, target_reconstructed)

        reconstruction_loss = F.mse_loss(target_reconstructed, bottlenecks[target])
        cyclic_loss = F.mse_loss(source_cyclic, bottlenecks[source])
        return reconstruction_loss + cyclic_loss, reconstruction_loss, cyclic_loss
    
    def _compute_translation_loss(self, bottlenecks):
        """
        Translation loss with forward reconstruction and cyclic consistency (Eq.20).
        """
        translation_loss = bottlenecks['t'].new_tensor(0.0)
        reconstruction_total = bottlenecks['t'].new_tensor(0.0)
        cyclic_total = bottlenecks['t'].new_tensor(0.0)

        for source, target in [('t', 'a'), ('a', 't'), ('t', 'v'), ('v', 't'), ('a', 'v'), ('v', 'a')]:
            pair_loss, reconstruction_loss, cyclic_loss = self._compute_directional_translation_loss(
                source, target, bottlenecks
            )
            translation_loss = translation_loss + pair_loss
            reconstruction_total = reconstruction_total + reconstruction_loss
            cyclic_total = cyclic_total + cyclic_loss

        return translation_loss, reconstruction_total, cyclic_total

    def _supplement_missing_latents(self, bottlenecks, modality_mask):
        supplemented = {}

        for target in self.modalities:
            target_idx = self.modality_to_idx[target]
            target_present = modality_mask[:, target_idx:target_idx + 1]

            reconstructed = torch.zeros_like(bottlenecks[target])
            num_sources = torch.zeros_like(target_present)

            for source in self.modalities:
                if source == target:
                    continue
                source_idx = self.modality_to_idx[source]
                source_present = modality_mask[:, source_idx:source_idx + 1]
                translated = self._translate_latent(source, target, bottlenecks[source])
                reconstructed = reconstructed + translated * source_present
                num_sources = num_sources + source_present

            fallback = bottlenecks[target]
            reconstructed = torch.where(num_sources > 0, reconstructed, fallback)
            supplemented[target] = target_present * bottlenecks[target] + (1.0 - target_present) * reconstructed

        return supplemented
    
    def forward(self, text, acoustic, visual, labels=None, stage=1, attention_mask=None, modality_mask=None):
        """
        Args:
            text: (batch, seq_len, 768) from DeBERTa
            acoustic: (batch, seq_len, 74)
            visual: (batch, seq_len, 47)
            labels: (batch,) for L_lib
            stage: 1 = no translation, 2 = with translation
            attention_mask: (batch, seq_len) valid-token mask
            modality_mask: (batch, 3) modality presence mask in t/a/v order
        
        Returns:
            logits: (batch, 1)
            ib_loss: scalar
            loss_dict: detailed losses
        """
        if attention_mask is None:
            token_mask = torch.ones(text.size(0), text.size(1), device=text.device, dtype=text.dtype)
        else:
            token_mask = attention_mask.to(text.device).float()
        modality_mask = self._prepare_modality_mask(modality_mask, text.size(0), text.device, text.dtype)
        
        if self.normalize_av:
            acoustic = F.normalize(acoustic, dim=-1)
            visual = F.normalize(visual, dim=-1)
        
        # ============ 模态投影 ============
        F_t = self.proj_t(text)  # (batch, seq_len, unified_dim)
        F_a = self.proj_a(acoustic)
        F_v = self.proj_v(visual)
        
        F = {'t': F_t, 'a': F_a, 'v': F_v}
        
        # ============ IB编码 ============
        B_t, mu_t, logvar_t = self.ib_enc_t(F_t)
        B_a, mu_a, logvar_a = self.ib_enc_a(F_a)
        B_v, mu_v, logvar_v = self.ib_enc_v(F_v)
        
        B = {'t': B_t, 'a': B_a, 'v': B_v}
        mu = {'t': mu_t, 'a': mu_a, 'v': mu_v}
        logvar = {'t': logvar_t, 'a': logvar_a, 'v': logvar_v}

        B_pooled = {m: self._masked_mean(B[m], token_mask) for m in ['t', 'a', 'v']}
        mu_pooled = {m: self._masked_mean(mu[m], token_mask) for m in ['t', 'a', 'v']}
        logvar_pooled = {m: self._masked_mean(logvar[m], token_mask) for m in ['t', 'a', 'v']}
        
        labels = self._prepare_labels(labels)

        # ============ Token-level IB Loss (Eq.9) ============
        L_tib = self._compute_cyclic_tib(F, B, mu, logvar, token_mask)
        
        # ============ Label-level IB Loss (Eq.11) ============
        if labels is not None:
            L_lib, label_loss_dict = self._compute_label_ib_loss(B_pooled, mu_pooled, logvar_pooled, labels)
        else:
            L_lib = torch.tensor(0.0, device=text.device)
            label_loss_dict = {}
        
        # ============ Translation Loss (Eq.20) ============
        complete_modalities = bool(torch.all(modality_mask > 0.5).item())
        if stage == 2 and labels is not None and complete_modalities:
            L_tran, L_rec, L_cyc = self._compute_translation_loss(B_pooled)
        else:
            L_tran = torch.tensor(0.0, device=text.device)
            L_rec = torch.tensor(0.0, device=text.device)
            L_cyc = torch.tensor(0.0, device=text.device)

        fusion_bottlenecks = self._supplement_missing_latents(B_pooled, modality_mask)
        B_t = fusion_bottlenecks['t']
        B_a = fusion_bottlenecks['a']
        B_v = fusion_bottlenecks['v']
        
        # ============ Total IB Loss (Eq.23) ============
        # L_total = L_task + (1/β) * (L_tib + L_lib) + γ * L_tran
        # 这里只返回IB部分，L_task在train.py中添加
        ib_loss = (1.0 / self.beta) * (L_tib + L_lib)
        if stage == 2:
            ib_loss = ib_loss + self.gamma * L_tran
        
        # ============ Multimodal Fusion (Eq.22) ============
        F_M = self.fusion(B_t, B_a, B_v)  # (batch, 3*bottleneck_dim)
        logits = self.predictor(F_M)
        
        # Loss详情
        loss_dict = {
            'L_tib': L_tib.item() if torch.is_tensor(L_tib) else L_tib,
            'L_lib': L_lib.item() if torch.is_tensor(L_lib) else L_lib,
            'L_tran': L_tran.item() if torch.is_tensor(L_tran) else L_tran,
            'L_rec': L_rec.item() if torch.is_tensor(L_rec) else L_rec,
            'L_cyc': L_cyc.item() if torch.is_tensor(L_cyc) else L_cyc,
        }
        loss_dict.update(label_loss_dict)
        
        return logits, ib_loss, loss_dict
