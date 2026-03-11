"""
CyIN 训练脚本
支持两阶段训练：
    Stage 1: 构建信息瓶颈空间 (gamma=0)
    Stage 2: 引入跨模态翻译 (gamma>0)
"""
import argparse
import os
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW
from deberta_CyIN import CyIN_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

# ============ 参数解析 ============
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosi_imagebind", "mosei", "iemocap", "meld"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=None, help="legacy fallback that applies to all parameter groups")
parser.add_argument("--encoder_learning_rate", type=float, default=4e-5, help="learning rate for the pretrained DeBERTa encoder")
parser.add_argument("--cyin_learning_rate", type=float, default=1e-3, help="learning rate for CyIN-specific modules")
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=128)

# CyIN specific parameters (Table 6 - MOSI defaults)
parser.add_argument('--unified_dim', default=256, type=int, help='C_U: unified projection dimension')
parser.add_argument('--ib_dim', default=256, type=int, help='C_ib: IB encoder/decoder dimension')
parser.add_argument('--bottleneck_dim', default=128, type=int, help='C_B: bottleneck latent dimension')
parser.add_argument('--drop_prob', default=0.3, type=float, help='dropout probability for CyIN')
parser.add_argument('--beta', default=32, type=float, help='IB trade-off coefficient (MOSI: 32)')
parser.add_argument('--gamma', default=10, type=float, help='translation loss coefficient (MOSI: 10)')
parser.add_argument('--cra_layers', default=8, type=int, help='number of CRA layers (MOSI: 8)')
parser.add_argument('--cra_layers_la', default=None, type=int, help='CRA layers for lang-audio translator (Table 4)')
parser.add_argument('--cra_layers_lv', default=None, type=int, help='CRA layers for lang-vision translator (Table 4)')
parser.add_argument('--cra_layers_av', default=None, type=int, help='CRA layers for audio-vision translator (Table 4)')
parser.add_argument('--cra_dims', default='64,32,16', type=str, help='CRA hidden dims (MOSI: 64,32,16)')
parser.add_argument('--attention_layers', default=2, type=int, help='cross-modal attention layers (MOSI: 2)')
parser.add_argument('--attention_heads', default=2, type=int, help='attention heads (MOSI: 2)')
# Two-stage training (1st:2nd = 1:9 for MOSI)
parser.add_argument('--stage1_epochs', default=5, type=int, help='epochs for stage 1 (no translation)')
parser.add_argument('--stage2_epochs', default=45, type=int, help='epochs for stage 2 (with translation)')
parser.add_argument('--stage2_missing_rate', default=0.5, type=float,
                    help='per-modality dropout probability when simulating incomplete inputs in stage 2')
parser.add_argument('--incomplete_task_weight', default=1.0, type=float,
                    help='weight for the task loss on simulated incomplete inputs during stage 2')
parser.add_argument('--checkpoint_prefix', default='cyIN_unified_refined', type=str,
                    help='prefix used when saving checkpoints under checkpoints/')

args = parser.parse_args()

# 处理cra_dims字符串转为列表
if isinstance(args.cra_dims, str):
    args.cra_dims = [int(x) for x in args.cra_dims.split(',')]

if args.learning_rate is not None:
    args.encoder_learning_rate = args.learning_rate
    args.cyin_learning_rate = args.learning_rate

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (
    global_configs.ACOUSTIC_DIM, 
    global_configs.VISUAL_DIM,
    global_configs.TEXT_DIM
)


# ============ 数据处理 ============
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        assert len(tokens) == len(inversions)

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

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

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


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def get_appropriate_dataset(data):
    tokenizer = get_tokenizer(args.model)
    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_label_ids,
    )
    return dataset


def _standardize_tensor(train_t, *other_ts):
    """Zero-mean unit-var standardization using train-set stats; skip padding (zeros)."""
    flat = train_t.reshape(-1, train_t.shape[-1])
    nonzero = flat.abs().sum(dim=-1) > 0
    if nonzero.sum() == 0:
        return (train_t,) + other_ts
    mean = flat[nonzero].mean(dim=0)
    std = flat[nonzero].std(dim=0).clamp(min=1e-6)
    results = []
    for t in (train_t,) + other_ts:
        mask = (t.abs().sum(dim=-1, keepdim=True) > 0).float()
        results.append((t - mean) * mask / std)
    print(f"  stats: mean_abs={mean.abs().mean():.4f}, std_mean={std.mean():.4f}")
    return tuple(results)


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step)
        * (args.stage1_epochs + args.stage2_epochs)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps


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
    print("Seed: {}".format(seed))


def build_optimizer_grouped_parameters(model: nn.Module):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    encoder_decay = []
    encoder_no_decay = []
    cyin_decay = []
    cyin_no_decay = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        is_encoder = name.startswith("dberta.model.")
        is_no_decay = any(nd in name for nd in no_decay)

        if is_encoder and is_no_decay:
            encoder_no_decay.append(parameter)
        elif is_encoder:
            encoder_decay.append(parameter)
        elif is_no_decay:
            cyin_no_decay.append(parameter)
        else:
            cyin_decay.append(parameter)

    optimizer_grouped_parameters = []
    for params, weight_decay, learning_rate in [
        (encoder_decay, 0.01, args.encoder_learning_rate),
        (encoder_no_decay, 0.0, args.encoder_learning_rate),
        (cyin_decay, 0.01, args.cyin_learning_rate),
        (cyin_no_decay, 0.0, args.cyin_learning_rate),
    ]:
        if params:
            optimizer_grouped_parameters.append(
                {
                    "params": params,
                    "weight_decay": weight_decay,
                    "lr": learning_rate,
                }
            )

    return optimizer_grouped_parameters


def prep_for_training(num_train_optimization_steps: int):
    model = CyIN_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )
    model.to(DEVICE)

    optimizer_grouped_parameters = build_optimizer_grouped_parameters(model)

    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def build_random_modality_mask(batch_size: int, device: torch.device, missing_rate: float):
    """Sample random missing-modality masks while ensuring at least one modality remains."""
    modality_mask = torch.ones(batch_size, 3, device=device)
    if missing_rate <= 0:
        return modality_mask

    drop_mask = torch.rand(batch_size, 3, device=device) < missing_rate
    all_missing = drop_mask.all(dim=1)
    if all_missing.any():
        keep_indices = torch.randint(0, 3, (int(all_missing.sum().item()),), device=device)
        drop_mask[all_missing] = True
        drop_mask[all_missing, keep_indices] = False

    modality_mask = (~drop_mask).float()
    return modality_mask


def compute_task_loss(logits: torch.Tensor, labels: torch.Tensor):
    return F.l1_loss(logits.view(-1), labels.view(-1))


def compute_total_objective(logits: torch.Tensor, labels: torch.Tensor, ib_loss: torch.Tensor):
    task_loss = compute_task_loss(logits, labels)
    return task_loss, task_loss + ib_loss


# ============ 训练函数 ============
def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler, stage=1):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Stage {stage}")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        label_ids = label_ids.reshape(label_ids.size(0), -1).squeeze(-1)

        visual_n = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        acoustic_n = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

        logits, ib_loss, loss_dict = model(
            input_ids,
            visual_n,
            acoustic_n,
            labels=label_ids,
            stage=stage,
        )

        task_loss, loss = compute_total_objective(logits, label_ids, ib_loss)

        if stage == 2 and args.stage2_missing_rate > 0:
            modality_mask = build_random_modality_mask(label_ids.size(0), DEVICE, args.stage2_missing_rate)
            incomplete_logits, _, _ = model(
                input_ids,
                visual_n,
                acoustic_n,
                stage=stage,
                modality_mask=modality_mask,
            )
            incomplete_task_loss = compute_task_loss(incomplete_logits, label_ids)
            loss = loss + args.incomplete_task_weight * incomplete_task_loss

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()
        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, stage=1):
    model.eval()
    dev_total_loss = 0
    dev_task_loss = 0
    nb_dev_steps = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            label_ids = label_ids.reshape(label_ids.size(0), -1).squeeze(-1)

            visual_n = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_n = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, ib_loss, loss_dict = model(
                input_ids,
                visual_n,
                acoustic_n,
                labels=label_ids,
                stage=stage,
            )

            task_loss, total_loss = compute_total_objective(logits, label_ids, ib_loss)

            if args.gradient_accumulation_step > 1:
                task_loss = task_loss / args.gradient_accumulation_step
                total_loss = total_loss / args.gradient_accumulation_step

            dev_task_loss += task_loss.item()
            dev_total_loss += total_loss.item()
            nb_dev_steps += 1

    return dev_total_loss / nb_dev_steps, dev_task_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader, stage=1):
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

            visual_n = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_n = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, ib_loss, loss_dict = model(
                input_ids,
                visual_n,
                acoustic_n,
                stage=stage,
            )

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, stage=1, use_zero=False):
    preds, y_test = test_epoch(model, test_dataloader, stage)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds_binary = preds >= 0
    y_test_binary = y_test >= 0

    f_score = f1_score(y_test_binary, preds_binary, average="weighted")
    acc = accuracy_score(y_test_binary, preds_binary)

    return acc, mae, corr, f_score


def train(model, train_dataloader, validation_dataloader, test_data_loader, optimizer, scheduler):
    """
    Two-stage training:
    - Stage 1: Build informative space (gamma=0 in CyIN forward)
    - Stage 2: Add translation training (gamma>0)
    """
    best_valid_loss = float('inf')
    best_checkpoint_path = os.path.join('checkpoints', f'{args.checkpoint_prefix}_best.pt')
    final_checkpoint_path = os.path.join('checkpoints', f'{args.checkpoint_prefix}_final.pt')

    os.makedirs('checkpoints', exist_ok=True)

    total_epochs = args.stage1_epochs + args.stage2_epochs

    for epoch_i in range(total_epochs):
        if epoch_i < args.stage1_epochs:
            stage = 1
            stage_epoch = epoch_i + 1
        else:
            stage = 2
            stage_epoch = epoch_i - args.stage1_epochs + 1

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, stage=stage)
        valid_total_loss, valid_task_loss = eval_epoch(model, validation_dataloader, stage=stage)

        test_acc, test_mae, test_corr, test_f_score = test_score_model(
            model, test_data_loader, stage=stage
        )
        print(
            f"Stage {stage} Epoch {stage_epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={valid_total_loss:.4f}, "
            f"Acc2={test_acc:.4f}, F1={test_f_score:.4f}, MAE={test_mae:.4f}, Corr={test_corr:.4f}"
        )

        should_track_best = stage == 2 or args.stage2_epochs == 0
        if should_track_best and valid_total_loss < best_valid_loss:
            best_valid_loss = valid_total_loss
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_loss': best_valid_loss,
                'args': args,
            }, best_checkpoint_path)
            print(f"  -> Saved best (val_loss={best_valid_loss:.4f})")

    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded best checkpoint from {best_checkpoint_path} for final evaluation")
    
    test_acc, test_mae, test_corr, test_f_score = test_score_model(model, test_data_loader, stage=2)
    print(f"\n{'='*60}")
    print(f"Final (best checkpoint): Acc2={test_acc:.4f}, F1={test_f_score:.4f}, MAE={test_mae:.4f}, Corr={test_corr:.4f}")
    print(f"Paper (Complete MOSI):   Acc2≈86.3%  F1≈86.3%  MAE=0.712  Corr=0.801")
    print(f"{'='*60}")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_test_acc': test_acc,
        'args': args,
    }, final_checkpoint_path)
    print(f"Saved final model to {final_checkpoint_path}")
    
    return test_acc, test_mae, test_corr, test_f_score


def main():
    set_random_seed(args.seed)
    effective_epochs = args.stage1_epochs + args.stage2_epochs
    
    print(f"Dataset: {args.dataset}")
    print(f"Training with CyIN (beta={args.beta}, gamma={args.gamma})")
    print(f"Stage 1 epochs: {args.stage1_epochs}, Stage 2 epochs: {args.stage2_epochs}")
    print(f"Learning rates: encoder={args.encoder_learning_rate}, cyin={args.cyin_learning_rate}")
    print(f"Stage 2 missing simulation: rate={args.stage2_missing_rate}, weight={args.incomplete_task_weight}")
    print(f"Checkpoint prefix: {args.checkpoint_prefix}")
    if args.n_epochs != effective_epochs:
        print(f"Ignoring n_epochs={args.n_epochs}; using stage total {effective_epochs} for scheduling and training.")
    
    train_data_loader, dev_data_loader, test_data_loader, num_train_optimization_steps = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
