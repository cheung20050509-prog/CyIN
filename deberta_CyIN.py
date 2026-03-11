"""
CyIN + DeBERTa 集成模块
使用 DeBERTa 作为文本编码器，并直接由 CyIN 的 MLP 头完成预测
"""
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from CyIN import CyIN
import global_configs
from global_configs import DEVICE


def _resolve_modality_dims(config, multimodal_config):
    text_dim = getattr(multimodal_config, 'text_dim', None) or global_configs.TEXT_DIM
    acoustic_dim = getattr(multimodal_config, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
    visual_dim = getattr(multimodal_config, 'visual_dim', None) or global_configs.VISUAL_DIM

    dataset_name = getattr(multimodal_config, 'dataset', None)
    if dataset_name and (text_dim <= 0 or acoustic_dim <= 0 or visual_dim <= 0):
        global_configs.set_dataset_config(dataset_name)
        text_dim = getattr(multimodal_config, 'text_dim', None) or global_configs.TEXT_DIM
        acoustic_dim = getattr(multimodal_config, 'acoustic_dim', None) or global_configs.ACOUSTIC_DIM
        visual_dim = getattr(multimodal_config, 'visual_dim', None) or global_configs.VISUAL_DIM

    if text_dim <= 0:
        text_dim = config.hidden_size

    if acoustic_dim <= 0 or visual_dim <= 0:
        raise ValueError(
            "Acoustic and visual feature dimensions are not configured. "
            "Call global_configs.set_dataset_config(...) before model creation or pass "
            "acoustic_dim/visual_dim in multimodal_config."
        )

    return text_dim, acoustic_dim, visual_dim


class CyIN_DebertaModel(DebertaV2PreTrainedModel):
    """
    CyIN + DeBERTa 编码器
    """
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = _resolve_modality_dims(config, multimodal_config)
        self.config = config
        
        # 加载预训练DeBERTa
        model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.model = model.to(DEVICE)
        
        # CyIN参数配置 (严格按Table 6)
        cyIN_args = {
            'text_dim': TEXT_DIM,
            'acoustic_dim': ACOUSTIC_DIM,
            'visual_dim': VISUAL_DIM,
            'unified_dim': getattr(multimodal_config, 'unified_dim', 256),
            'ib_dim': getattr(multimodal_config, 'ib_dim', 256),
            'bottleneck_dim': getattr(multimodal_config, 'bottleneck_dim', 128),
            'dropout_prob': getattr(multimodal_config, 'drop_prob', 0.3),
            'beta': getattr(multimodal_config, 'beta', 32),
            'gamma': getattr(multimodal_config, 'gamma', 10),
            'cra_layers': getattr(multimodal_config, 'cra_layers', 8),
            'cra_dims': getattr(multimodal_config, 'cra_dims', [64, 32, 16]),
            'attention_layers': getattr(multimodal_config, 'attention_layers', 2),
            'attention_heads': getattr(multimodal_config, 'attention_heads', 2),
        }
        
        self.CyIN = CyIN(cyIN_args)
        self.init_weights()

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            labels=None,
            stage=1,
            modality_mask=None,
    ):
        """
        Args:
            input_ids: (batch, seq_len) 文本token IDs
            visual: (batch, seq_len, visual_dim) 视觉特征
            acoustic: (batch, seq_len, acoustic_dim) 音频特征
            labels: (batch,) 标签 (用于label-level IB)
            stage: 训练阶段 (1=informative space, 2=with translation)
        """
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        attention_mask = input_ids.ne(pad_token_id).long()

        # DeBERTa编码文本
        embedding_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = embedding_output[0]  # (batch, seq_len, 768)

        if modality_mask is None:
            modality_mask = text_features.new_ones(text_features.size(0), 3)
        else:
            modality_mask = modality_mask.to(device=text_features.device, dtype=text_features.dtype)

        text_features = text_features * modality_mask[:, 0].view(-1, 1, 1)
        acoustic = acoustic * modality_mask[:, 1].view(-1, 1, 1)
        visual = visual * modality_mask[:, 2].view(-1, 1, 1)

        logits, ib_loss, loss_dict = self.CyIN(
            text_features, acoustic, visual, 
            labels=labels, stage=stage, attention_mask=attention_mask, modality_mask=modality_mask
        )

        return logits, ib_loss, loss_dict


class CyIN_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    """
    CyIN + DeBERTa 序列分类/回归模型
    """
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.dberta = CyIN_DebertaModel(config, multimodal_config)

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            labels=None,
            stage=1,
            modality_mask=None,
    ):
        logits, ib_loss, loss_dict = self.dberta(
            input_ids,
            visual,
            acoustic,
            labels=labels,
            stage=stage,
            modality_mask=modality_mask,
        )

        return logits, ib_loss, loss_dict
