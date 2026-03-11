"""
CyIN 测试脚本
加载训练好的模型并在测试集上评估
支持完整模态和缺失模态测试
"""
import argparse
import os
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import DebertaV2Tokenizer
from deberta_CyIN import CyIN_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

# ============ 参数解析 ============
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosi_imagebind", "mosei", "iemocap", "meld"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=128)

# CyIN参数 (需要与训练时一致 - Table 6 MOSI)
parser.add_argument('--unified_dim', default=256, type=int, help='C_U')
parser.add_argument('--ib_dim', default=256, type=int, help='C_ib')
parser.add_argument('--bottleneck_dim', default=128, type=int, help='C_B')
parser.add_argument('--drop_prob', default=0.3, type=float)
parser.add_argument('--dropout_prob', default=0.1, type=float)
parser.add_argument('--beta', default=32, type=float)
parser.add_argument('--gamma', default=10, type=float)
parser.add_argument('--cra_layers', default=8, type=int)
parser.add_argument('--cra_dims', default='64,32,16', type=str)
parser.add_argument('--attention_layers', default=2, type=int)
parser.add_argument('--attention_heads', default=2, type=int)
# 模型路径
parser.add_argument('--checkpoint', type=str, default='checkpoints/cyIN_unified_refined_best.pt', 
                    help='path to trained model checkpoint')
parser.add_argument('--complete_only', action='store_true',
                    help='only evaluate the complete-modality setting and skip all missing-modality reports')

# 缺失模态测试
parser.add_argument('--missing_rate', type=float, default=0.0, 
                    help='random missing rate (0.0-1.0)')
parser.add_argument('--missing_modality', type=str, default=None, 
                    choices=['text', 'acoustic', 'visual', 'ta', 'tv', 'av'],
                    help='fixed missing modality')

args = parser.parse_args()

# 处理cra_dims字符串转为列表
if isinstance(args.cra_dims, str):
    args.cra_dims = [int(x) for x in args.cra_dims.split(',')]

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (
    global_configs.ACOUSTIC_DIM, 
    global_configs.VISUAL_DIM,
    global_configs.TEXT_DIM
)


# ============ 数据处理 ============
class InputFeatures(object):
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        aligned_visual = []
        aligned_audio = []
        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_deberta_input(
            tokens, visual, acoustic, tokenizer
        )

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def get_test_dataset():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)
    
    test_data = data["test"]
    tokenizer = get_tokenizer(args.model)
    features = convert_to_features(test_data, args.max_seq_length, tokenizer)
    
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_visual, all_acoustic, all_label_ids)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)
    
    return dataloader


def set_random_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path):
    """加载训练好的模型"""
    model = CyIN_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        legacy_keys = [
            key for key in state_dict
            if key.startswith('dberta.pooler.')
            or key.startswith('dberta.expand.')
            or key.startswith('dberta.LayerNorm.')
            or key.startswith('dberta.dropout.')
            or key.startswith('classifier.')
            or key == 'dropout.p'
        ]
        if legacy_keys:
            raise RuntimeError(
                'Legacy residual-wrapper checkpoint detected. '
                'The model now predicts directly from CyIN F_M -> predictor, '
                'so old checkpoints must be retrained with the current architecture.'
            )
        pre_refine_cra_keys = [
            key for key in state_dict
            if '.CyIN.cra_' in key
        ]
        if pre_refine_cra_keys:
            raise RuntimeError(
                'Pre-refinement CyIN checkpoint detected. '
                'The current model shares CRA translators across cyclic directions '
                'and uses the refined fusion block, so this checkpoint must be retrained.'
            )
        missing = [k for k in model.state_dict() if k not in state_dict]
        if missing:
            print(f"  Note: {len(missing)} keys not in checkpoint (architecture updated) — loading with strict=False")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using pretrained weights only")
    
    model.to(DEVICE)
    model.eval()
    return model


def build_modality_mask(batch_size, device, missing_modality=None, missing_rate=0.0):
    """
    构建模态存在掩码，顺序为 text / acoustic / visual。
    """
    modality_mask = torch.ones(batch_size, 3, device=device)

    fixed_missing_map = {
        'text': ('t',),
        'acoustic': ('a',),
        'visual': ('v',),
        'ta': ('t', 'a'),
        'tv': ('t', 'v'),
        'av': ('a', 'v'),
    }
    modality_index = {'t': 0, 'a': 1, 'v': 2}

    if missing_modality is not None:
        for modality in fixed_missing_map[missing_modality]:
            modality_mask[:, modality_index[modality]] = 0.0
    elif missing_rate > 0.0:
        random_patterns = [
            ('t',),
            ('a',),
            ('v',),
            ('t', 'a'),
            ('t', 'v'),
            ('a', 'v'),
        ]
        for i in range(batch_size):
            if random.random() < missing_rate:
                missing_pattern = random.choice(random_patterns)
                for modality in missing_pattern:
                    modality_mask[i, modality_index[modality]] = 0.0

    return modality_mask


def test_model(model, test_dataloader, missing_modality=None, missing_rate=0.0):
    """测试模型"""
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            label_ids = label_ids.reshape(label_ids.size(0), -1)
            modality_mask = build_modality_mask(
                input_ids.size(0),
                input_ids.device,
                missing_modality=missing_modality,
                missing_rate=missing_rate,
            )

            # 归一化
            visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, ib_loss, loss_dict = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                stage=2,  # 使用完整模型包括翻译
                modality_mask=modality_mask,
            )

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            if isinstance(logits, float):
                logits = [logits]
                label_ids = [label_ids]

            preds.extend(logits)
            labels.extend(label_ids)

    preds = np.array(preds)
    labels = np.array(labels)
    
    return preds, labels


def compute_metrics(preds, labels, use_zero=False):
    """计算评估指标"""
    non_zeros = np.array([i for i, e in enumerate(labels) if e != 0 or use_zero])
    
    preds_filtered = preds[non_zeros]
    labels_filtered = labels[non_zeros]

    # 回归指标
    mae = np.mean(np.absolute(preds_filtered - labels_filtered))
    corr = np.corrcoef(preds_filtered, labels_filtered)[0][1]

    # 分类指标 (二分类: positive/negative)
    preds_binary = preds_filtered >= 0
    labels_binary = labels_filtered >= 0
    
    acc2 = accuracy_score(labels_binary, preds_binary)
    f1 = f1_score(labels_binary, preds_binary, average="weighted")

    # 7分类 (sentiment levels)
    preds_7 = np.clip(np.round(preds_filtered), -3, 3).astype(int)
    labels_7 = np.clip(np.round(labels_filtered), -3, 3).astype(int)
    acc7 = accuracy_score(labels_7, preds_7)

    return {
        'MAE': mae,
        'Corr': corr,
        'Acc2': acc2,
        'Acc7': acc7,
        'F1': f1,
    }


def main():
    set_random_seed(args.seed)
    
    print("=" * 60)
    print(f"CyIN Test Script")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)
    
    # 加载模型
    model = load_model(args.checkpoint)
    
    # 加载测试集
    test_dataloader = get_test_dataset()
    print(f"Test samples: {len(test_dataloader.dataset)}")
    
    # 完整模态测试
    print("\n[Complete Modality Test]")
    preds, labels = test_model(model, test_dataloader)
    complete_metrics = compute_metrics(preds, labels)
    print(f"  Acc2: {complete_metrics['Acc2']:.4f}")
    print(f"  Acc7: {complete_metrics['Acc7']:.4f}")
    print(f"  F1:   {complete_metrics['F1']:.4f}")
    print(f"  MAE:  {complete_metrics['MAE']:.4f}")
    print(f"  Corr: {complete_metrics['Corr']:.4f}")

    if args.complete_only:
        return
    
    # 固定缺失模态测试
    if args.missing_modality:
        print(f"\n[Missing Modality: {args.missing_modality}]")
        preds, labels = test_model(model, test_dataloader, missing_modality=args.missing_modality)
        metrics = compute_metrics(preds, labels)
        print(f"  Acc2: {metrics['Acc2']:.4f}")
        print(f"  Acc7: {metrics['Acc7']:.4f}")
        print(f"  F1:   {metrics['F1']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  Corr: {metrics['Corr']:.4f}")
    
    # 随机缺失测试
    if args.missing_rate > 0:
        print(f"\n[Random Missing Rate: {args.missing_rate}]")
        preds, labels = test_model(model, test_dataloader, missing_rate=args.missing_rate)
        metrics = compute_metrics(preds, labels)
        print(f"  Acc2: {metrics['Acc2']:.4f}")
        print(f"  Acc7: {metrics['Acc7']:.4f}")
        print(f"  F1:   {metrics['F1']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  Corr: {metrics['Corr']:.4f}")
    
    # 全面缺失模态测试 (如果没有指定特定缺失)
    if not args.missing_modality and args.missing_rate == 0:
        print("\n" + "=" * 60)
        print("Paper Protocol: Fixed Missing (6 configs avg)")
        print("=" * 60)
        
        # 论文 Fixed Missing: 6种组合的平均
        # 保留2个模态（缺1个）: {l,a}缺v, {l,v}缺a, {a,v}缺t  
        # 保留1个模态（缺2个）: {l}缺av, {a}缺tv, {v}缺ta
        fixed_missing_configs = [
            ('visual', 'u={l,a} (miss v)'),      # 保留 l,a
            ('acoustic', 'u={l,v} (miss a)'),   # 保留 l,v  
            ('text', 'u={a,v} (miss t)'),       # 保留 a,v
            ('av', 'u={l} (miss a,v)'),         # 仅保留 l
            ('tv', 'u={a} (miss t,v)'),         # 仅保留 a
            ('ta', 'u={v} (miss t,a)'),         # 仅保留 v
        ]
        
        fixed_acc2_list = []
        fixed_f1_list = []
        fixed_mae_list = []
        fixed_corr_list = []
        
        for missing_mod, desc in fixed_missing_configs:
            preds, labels = test_model(model, test_dataloader, missing_modality=missing_mod)
            metrics = compute_metrics(preds, labels)
            fixed_acc2_list.append(metrics['Acc2'])
            fixed_f1_list.append(metrics['F1'])
            fixed_mae_list.append(metrics['MAE'])
            fixed_corr_list.append(metrics['Corr'])
            print(f"  {desc}: Acc2={metrics['Acc2']:.4f}, F1={metrics['F1']:.4f}, MAE={metrics['MAE']:.4f}")
        
        avg_fixed_acc2 = np.mean(fixed_acc2_list)
        avg_fixed_f1 = np.mean(fixed_f1_list)
        avg_fixed_mae = np.mean(fixed_mae_list)
        avg_fixed_corr = np.mean(fixed_corr_list)
        print(f"\n  ** Fixed Missing AVG: Acc2={avg_fixed_acc2:.4f}, F1={avg_fixed_f1:.4f}, MAE={avg_fixed_mae:.4f}, Corr={avg_fixed_corr:.4f}")
        
        print("\n" + "=" * 60)
        print("Paper Protocol: Random Missing (7 MRs avg)")
        print("=" * 60)
        
        # 论文 Random Missing: MR ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7} 的平均
        random_mrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        random_acc2_list = []
        random_f1_list = []
        random_mae_list = []
        random_corr_list = []
        
        for mr in random_mrs:
            preds, labels = test_model(model, test_dataloader, missing_rate=mr)
            metrics = compute_metrics(preds, labels)
            random_acc2_list.append(metrics['Acc2'])
            random_f1_list.append(metrics['F1'])
            random_mae_list.append(metrics['MAE'])
            random_corr_list.append(metrics['Corr'])
            print(f"  MR={mr}: Acc2={metrics['Acc2']:.4f}, F1={metrics['F1']:.4f}, MAE={metrics['MAE']:.4f}")
        
        avg_random_acc2 = np.mean(random_acc2_list)
        avg_random_f1 = np.mean(random_f1_list)
        avg_random_mae = np.mean(random_mae_list)
        avg_random_corr = np.mean(random_corr_list)
        print(f"\n  ** Random Missing AVG: Acc2={avg_random_acc2:.4f}, F1={avg_random_f1:.4f}, MAE={avg_random_mae:.4f}, Corr={avg_random_corr:.4f}")
        
        # 最终对比表格
        print("\n" + "=" * 60)
        print("Summary vs Paper (MOSI)")
        print("=" * 60)
        print(f"{'Setting':<20} {'Acc2':>10} {'F1':>10} {'MAE':>10} {'Corr':>10}")
        print("-" * 60)
        print(f"{'Complete (Ours)':<20} {complete_metrics['Acc2']*100:>9.1f}% {complete_metrics['F1']*100:>9.1f}% {complete_metrics['MAE']:>10.3f} {complete_metrics['Corr']:>10.3f}")
        print(f"{'Complete (Paper)':<20} {'86.1':>10} {'53.2':>10} {'0.712':>10} {'0.801':>10}")
        print(f"{'Fixed Missing (Ours)':<20} {avg_fixed_acc2*100:>9.1f}% {avg_fixed_f1*100:>9.1f}% {avg_fixed_mae:>10.3f} {avg_fixed_corr:>10.3f}")
        print(f"{'Fixed Missing (Paper)':<20} {'78.6':>10} {'47.6':>10} {'1.037':>10} {'0.599':>10}")
        print(f"{'Random Missing (Ours)':<20} {avg_random_acc2*100:>9.1f}% {avg_random_f1*100:>9.1f}% {avg_random_mae:>10.3f} {avg_random_corr:>10.3f}")
        print(f"{'Random Missing (Paper)':<20} {'79.9':>10} {'48.3':>10} {'0.943':>10} {'0.650':>10}")


if __name__ == "__main__":
    main()
