#!/usr/bin/env python
"""
Extract ImageBind features for CMU-MOSI dataset.
Replaces original acoustic (74-dim) and visual (47-dim) features 
with ImageBind embeddings (1024-dim each).
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# Add ImageBind to path
IMAGEBIND_PATH = Path(__file__).parent.parent / "ImageBind"
sys.path.insert(0, str(IMAGEBIND_PATH))

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mosi_pkl', type=str, 
                        default='datasets/mosi.pkl',
                        help='Path to original MOSI pickle')
    parser.add_argument('--raw_audio_dir', type=str,
                        default='/home/datasets/mosi/raw/audio',
                        help='Directory containing raw .wav files')
    parser.add_argument('--raw_video_dir', type=str,
                        default='/home/datasets/mosi/raw/video',
                        help='Directory containing raw .mp4 files')
    parser.add_argument('--output_pkl', type=str,
                        default='datasets/mosi_imagebind.pkl',
                        help='Output pickle with ImageBind features')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for feature extraction')
    return parser.parse_args()


def load_imagebind_model(device):
    """Load pretrained ImageBind model."""
    print("Loading ImageBind model...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print("ImageBind model loaded successfully.")
    return model


def parse_segment_id(segment):
    """Parse segment string like '03bSnISJMiM[0]' to (video_id, segment_num)."""
    # Format: video_id[segment_num]
    if '[' in segment:
        video_id = segment.split('[')[0]
        segment_num = segment.split('[')[1].rstrip(']')
    else:
        # Alternative format: video_id_segment_num
        parts = segment.rsplit('_', 1)
        video_id = parts[0]
        segment_num = parts[1] if len(parts) > 1 else '0'
    return video_id, segment_num


def extract_features_batch(model, audio_paths, video_paths, device):
    """Extract ImageBind features for a batch of audio/video pairs."""
    
    # Prepare inputs for ImageBind
    inputs = {}
    
    # Filter valid paths
    valid_audio = [p for p in audio_paths if os.path.exists(p)]
    valid_video = [p for p in video_paths if os.path.exists(p)]
    
    if valid_audio:
        inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data(
            valid_audio, device
        )
    
    if valid_video:
        inputs[ModalityType.VISION] = data.load_and_transform_video_data(
            valid_video, device
        )
    
    if not inputs:
        return None, None
    
    with torch.no_grad():
        embeddings = model(inputs)
    
    audio_feats = embeddings.get(ModalityType.AUDIO, None)
    video_feats = embeddings.get(ModalityType.VISION, None)
    
    if audio_feats is not None:
        audio_feats = audio_feats.cpu().numpy()
    if video_feats is not None:
        video_feats = video_feats.cpu().numpy()
    
    return audio_feats, video_feats


def process_dataset_split(model, samples, raw_audio_dir, raw_video_dir, device, batch_size=16):
    """Process all samples in a dataset split."""
    
    new_samples = []
    failed_samples = []
    
    # Collect all segment info first
    segments_info = []
    for sample in samples:
        (words, visual, acoustic), label, segment = sample
        video_id, segment_num = parse_segment_id(segment)
        
        # Construct file paths
        # Try different naming conventions
        audio_path = os.path.join(raw_audio_dir, f"{video_id}_{segment_num}.wav")
        video_path = os.path.join(raw_video_dir, f"{video_id}_{segment_num}.mp4")
        
        # Fallback: check if segment_num needs adjustment (0-indexed vs 1-indexed)
        if not os.path.exists(audio_path):
            alt_num = str(int(segment_num) + 1)
            audio_path = os.path.join(raw_audio_dir, f"{video_id}_{alt_num}.wav")
            video_path = os.path.join(raw_video_dir, f"{video_id}_{alt_num}.mp4")
        
        segments_info.append({
            'sample': sample,
            'audio_path': audio_path,
            'video_path': video_path,
            'segment': segment
        })
    
    # Process in batches
    for i in tqdm(range(0, len(segments_info), batch_size), desc="Extracting features"):
        batch = segments_info[i:i+batch_size]
        
        audio_paths = [b['audio_path'] for b in batch]
        video_paths = [b['video_path'] for b in batch]
        
        try:
            audio_feats, video_feats = extract_features_batch(
                model, audio_paths, video_paths, device
            )
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Process individually for failed batch
            for b in batch:
                (words, visual, acoustic), label, segment = b['sample']
                # Keep original features if extraction fails
                new_samples.append(b['sample'])
                failed_samples.append(segment)
            continue
        
        # Assign features to samples
        for j, b in enumerate(batch):
            (words, visual, acoustic), label, segment = b['sample']
            seq_len = len(words)
            
            # Get features (or fallback to zeros)
            if audio_feats is not None and j < len(audio_feats):
                # Expand 1024-dim to sequence length (repeat for each word)
                new_acoustic = np.tile(audio_feats[j], (seq_len, 1))
            else:
                new_acoustic = np.zeros((seq_len, 1024), dtype=np.float32)
                failed_samples.append(segment)
            
            if video_feats is not None and j < len(video_feats):
                new_visual = np.tile(video_feats[j], (seq_len, 1))
            else:
                new_visual = np.zeros((seq_len, 1024), dtype=np.float32)
                if segment not in failed_samples:
                    failed_samples.append(segment)
            
            new_sample = ((words, new_visual, new_acoustic), label, segment)
            new_samples.append(new_sample)
    
    return new_samples, failed_samples


def main():
    args = parse_args()
    
    # Load original dataset
    print(f"Loading original MOSI dataset from {args.mosi_pkl}...")
    with open(args.mosi_pkl, 'rb') as f:
        original_data = pickle.load(f)
    
    print(f"Train: {len(original_data['train'])} samples")
    print(f"Dev: {len(original_data['dev'])} samples")  
    print(f"Test: {len(original_data['test'])} samples")
    
    # Load ImageBind model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_imagebind_model(device)
    
    # Process each split
    new_data = {}
    all_failed = []
    
    for split in ['train', 'dev', 'test']:
        print(f"\nProcessing {split} split...")
        new_samples, failed = process_dataset_split(
            model, 
            original_data[split],
            args.raw_audio_dir,
            args.raw_video_dir,
            device,
            batch_size=args.batch_size
        )
        new_data[split] = new_samples
        all_failed.extend(failed)
        print(f"  Processed {len(new_samples)} samples, {len(failed)} failed")
    
    # Save new dataset
    output_dir = os.path.dirname(args.output_pkl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving ImageBind features to {args.output_pkl}...")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(new_data, f)
    
    # Print summary
    print("\n" + "="*60)
    print("Feature Extraction Complete!")
    print("="*60)
    print(f"Original feature dims: visual=47, acoustic=74")
    print(f"ImageBind feature dims: visual=1024, acoustic=1024")
    print(f"Total samples: {sum(len(new_data[s]) for s in new_data)}")
    print(f"Failed extractions: {len(set(all_failed))}")
    print(f"Output saved to: {args.output_pkl}")
    
    if all_failed:
        print(f"\nFailed segments (first 10): {all_failed[:10]}")


if __name__ == "__main__":
    main()
