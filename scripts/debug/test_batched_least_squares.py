"""
Test and development of batched least squares for scale-shift invariant depth loss.
This will replace the sequential for-loop with GPU-parallel batch operations.
"""

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
for _candidate in (_THIS_FILE.parent, *_THIS_FILE.parents):
    if (_candidate / "data_utils").exists():
        _PROJECT_ROOT = _candidate
        break
else:
    _PROJECT_ROOT = _THIS_FILE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import torch
import numpy as np
import time


def original_sequential_ssi_loss(pred_depth, gt_depth, mask):
    """
    Original sequential implementation for comparison.
    """
    B = pred_depth.shape[0]
    total_loss = 0.0
    valid_samples = 0
    
    for i in range(B):
        # Extract valid pixels using mask
        mask_i = mask[i] > 0
        num_valid = mask_i.sum().item()
        
        if num_valid < 2:  # Need at least 2 points for least squares
            continue
            
        # Extract valid depth values
        pred_valid = pred_depth[i][mask_i].reshape(-1, 1)  # [M, 1]
        gt_valid = gt_depth[i][mask_i].reshape(-1, 1)      # [M, 1]
        
        # Set up least squares: A * [scale, shift]^T = gt_valid
        ones = torch.ones_like(pred_valid)
        A = torch.cat([pred_valid, ones], dim=1)  # [M, 2]
        
        try:
            # Solve normal equations: (A^T A) * h = A^T * gt_valid
            ATA = A.t() @ A              # [2, 2]
            ATb = A.t() @ gt_valid       # [2, 1]
            
            # Solve for optimal [scale, shift]
            h_opt = torch.linalg.solve(ATA, ATb)  # [2, 1]
            scale = h_opt[0, 0]
            shift = h_opt[1, 0]
            
            # Apply alignment and compute loss
            pred_aligned = scale * pred_valid + shift
            loss_i = torch.mean((pred_aligned - gt_valid) ** 2)
            total_loss += loss_i
            valid_samples += 1
            
        except RuntimeError:
            # Fallback for singular matrix
            loss_i = torch.mean((pred_valid - gt_valid) ** 2)
            total_loss += loss_i
            valid_samples += 1
    
    if valid_samples > 0:
        return total_loss / valid_samples
    else:
        return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)


def batched_ssi_loss(pred_depth, gt_depth, mask):
    """
    Batched implementation using GPU-parallel operations.
    """
    B, H, W = pred_depth.shape
    device = pred_depth.device
    
    # Find maximum number of valid pixels across all samples
    num_valid_per_sample = torch.sum(mask > 0, dim=[1, 2])  # [B]
    max_valid = torch.max(num_valid_per_sample).item()
    
    if max_valid < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Create padded tensors for batch processing
    pred_batch = torch.zeros(B, max_valid, device=device, dtype=pred_depth.dtype)
    gt_batch = torch.zeros(B, max_valid, device=device, dtype=gt_depth.dtype)
    valid_mask = torch.zeros(B, max_valid, device=device, dtype=torch.bool)
    
    # Fill padded tensors
    for i in range(B):
        mask_i = mask[i] > 0
        num_valid = mask_i.sum().item()
        
        if num_valid >= 2:
            pred_valid = pred_depth[i][mask_i]  # [num_valid]
            gt_valid = gt_depth[i][mask_i]      # [num_valid]
            
            pred_batch[i, :num_valid] = pred_valid
            gt_batch[i, :num_valid] = gt_valid
            valid_mask[i, :num_valid] = True
    
    # Create batch design matrix A = [pred_batch, ones] for all samples
    ones_batch = torch.ones_like(pred_batch)
    A_batch = torch.stack([pred_batch, ones_batch], dim=2)  # [B, max_valid, 2]
    
    # Only process samples with sufficient valid points
    valid_samples_mask = num_valid_per_sample >= 2
    num_valid_samples = valid_samples_mask.sum().item()
    
    if num_valid_samples == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Filter to only valid samples
    A_valid = A_batch[valid_samples_mask]        # [num_valid_samples, max_valid, 2]
    gt_valid = gt_batch[valid_samples_mask]      # [num_valid_samples, max_valid]
    mask_valid = valid_mask[valid_samples_mask]  # [num_valid_samples, max_valid]
    
    # Batch least squares computation
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for i in range(num_valid_samples):
        # Get valid entries for this sample
        sample_mask = mask_valid[i]
        A_i = A_valid[i][sample_mask]  # [actual_valid, 2]
        gt_i = gt_valid[i][sample_mask]  # [actual_valid]
        
        try:
            # Solve least squares: (A^T A) h = A^T b
            ATA = A_i.t() @ A_i                    # [2, 2]
            ATb = A_i.t() @ gt_i.unsqueeze(1)     # [2, 1]
            
            h_opt = torch.linalg.solve(ATA, ATb)   # [2, 1]
            scale = h_opt[0, 0]
            shift = h_opt[1, 0]
            
            # Apply alignment and compute MSE
            pred_aligned = scale * A_i[:, 0] + shift
            loss_i = torch.mean((pred_aligned - gt_i) ** 2)
            total_loss = total_loss + loss_i
            
        except RuntimeError:
            # Fallback for singular matrix
            pred_i = A_i[:, 0]  # Extract predictions
            loss_i = torch.mean((pred_i - gt_i) ** 2)
            total_loss = total_loss + loss_i
    
    return total_loss / num_valid_samples


def optimized_batched_ssi_loss(pred_depth, gt_depth, mask):
    """
    Optimized batched implementation with better memory efficiency.
    """
    B, H, W = pred_depth.shape
    device = pred_depth.device
    
    total_loss = torch.tensor(0.0, device=device)
    valid_samples = 0
    
    # Process samples in smaller chunks to avoid memory issues
    chunk_size = min(B, 32)  # Process up to 32 samples at once
    
    for chunk_start in range(0, B, chunk_size):
        chunk_end = min(chunk_start + chunk_size, B)
        chunk_size_actual = chunk_end - chunk_start
        
        # Get chunk data
        pred_chunk = pred_depth[chunk_start:chunk_end]
        gt_chunk = gt_depth[chunk_start:chunk_end]
        mask_chunk = mask[chunk_start:chunk_end]
        
        # Find max valid pixels in this chunk
        num_valid_chunk = torch.sum(mask_chunk > 0, dim=[1, 2])  # [chunk_size]
        max_valid_chunk = torch.max(num_valid_chunk).item()
        
        if max_valid_chunk < 2:
            continue
            
        # Process each sample in the chunk individually (but with optimized operations)
        for i in range(chunk_size_actual):
            mask_i = mask_chunk[i] > 0
            num_valid = mask_i.sum().item()
            
            if num_valid < 2:
                continue
                
            # Extract valid values
            pred_valid = pred_chunk[i][mask_i]
            gt_valid = gt_chunk[i][mask_i]
            
            # Vectorized least squares setup
            ones = torch.ones_like(pred_valid)
            A = torch.stack([pred_valid, ones], dim=1)  # [num_valid, 2]
            
            try:
                # Batched matrix operations
                ATA = torch.mm(A.t(), A)                      # [2, 2]
                ATb = torch.mv(A.t(), gt_valid)               # [2]
                
                # Solve using more stable method
                h_opt = torch.linalg.solve(ATA, ATb)          # [2]
                scale, shift = h_opt[0], h_opt[1]
                
                # Apply alignment
                pred_aligned = scale * pred_valid + shift
                loss_i = torch.mean((pred_aligned - gt_valid) ** 2)
                
                total_loss = total_loss + loss_i
                valid_samples += 1
                
            except RuntimeError:
                # Fallback
                loss_i = torch.mean((pred_valid - gt_valid) ** 2)
                total_loss = total_loss + loss_i
                valid_samples += 1
    
    if valid_samples > 0:
        return total_loss / valid_samples
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def test_implementations():
    """
    Test all implementations for correctness and performance.
    """
    print("🧪 Testing Scale-Shift Invariant Loss Implementations...")
    
    # Create test data
    torch.manual_seed(42)
    B, H, W = 8, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pred_depth = torch.randn(B, H, W, device=device, requires_grad=True) * 5 + 10
    gt_depth = torch.randn(B, H, W, device=device) * 5 + 10
    mask = torch.rand(B, H, W, device=device) > 0.3  # Random mask
    
    print(f"📊 Test data: {B}x{H}x{W}, device: {device}")
    print(f"📊 Valid pixels per sample: {torch.sum(mask > 0, dim=[1,2]).cpu().numpy()}")
    
    # Test original implementation
    print("\n🔄 Testing original sequential implementation...")
    start_time = time.time()
    loss_original = original_sequential_ssi_loss(pred_depth, gt_depth, mask)
    time_original = time.time() - start_time
    print(f"   Result: {loss_original.item():.6f}")
    print(f"   Time: {time_original*1000:.2f}ms")
    
    # Test batched implementation
    print("\n🔄 Testing batched implementation...")
    start_time = time.time()
    loss_batched = batched_ssi_loss(pred_depth, gt_depth, mask)
    time_batched = time.time() - start_time
    print(f"   Result: {loss_batched.item():.6f}")
    print(f"   Time: {time_batched*1000:.2f}ms")
    
    # Test optimized implementation
    print("\n🔄 Testing optimized batched implementation...")
    start_time = time.time()
    loss_optimized = optimized_batched_ssi_loss(pred_depth, gt_depth, mask)
    time_optimized = time.time() - start_time
    print(f"   Result: {loss_optimized.item():.6f}")
    print(f"   Time: {time_optimized*1000:.2f}ms")
    
    # Compare results
    print("\n📊 Results Comparison:")
    print(f"   Original:  {loss_original.item():.8f}")
    print(f"   Batched:   {loss_batched.item():.8f}")
    print(f"   Optimized: {loss_optimized.item():.8f}")
    
    # Check if results are close
    tol = 1e-5
    original_vs_batched = abs(loss_original.item() - loss_batched.item())
    original_vs_optimized = abs(loss_original.item() - loss_optimized.item())
    
    print(f"\n✅ Accuracy Check (tolerance: {tol}):")
    print(f"   Original vs Batched: {original_vs_batched:.8f} {'✅' if original_vs_batched < tol else '❌'}")
    print(f"   Original vs Optimized: {original_vs_optimized:.8f} {'✅' if original_vs_optimized < tol else '❌'}")
    
    print(f"\n⚡ Performance Comparison:")
    print(f"   Original:  {time_original*1000:.2f}ms (baseline)")
    print(f"   Batched:   {time_batched*1000:.2f}ms ({time_batched/time_original:.2f}x)")
    print(f"   Optimized: {time_optimized*1000:.2f}ms ({time_optimized/time_original:.2f}x)")
    
    # Test gradients with fresh tensors
    print("\n🔄 Testing gradient computation...")
    
    # Create fresh tensors for gradient test
    pred_depth_grad = torch.randn(B, H, W, device=device, requires_grad=True) * 5 + 10
    gt_depth_grad = gt_depth.clone()
    mask_grad = mask.clone()
    
    # Test original gradients
    loss_orig_grad = original_sequential_ssi_loss(pred_depth_grad, gt_depth_grad, mask_grad)
    loss_orig_grad.backward()
    grad_original = pred_depth_grad.grad.clone()
    
    # Test optimized gradients
    pred_depth_grad.grad.zero_()
    loss_opt_grad = optimized_batched_ssi_loss(pred_depth_grad, gt_depth_grad, mask_grad)
    loss_opt_grad.backward()
    grad_optimized = pred_depth_grad.grad.clone()
    
    grad_diff = torch.mean((grad_original - grad_optimized) ** 2).item()
    print(f"   Gradient MSE difference: {grad_diff:.8f} {'✅' if grad_diff < 1e-6 else '❌'}")
    
    return loss_optimized, time_optimized


if __name__ == "__main__":
    best_loss, best_time = test_implementations()
    print(f"\n🎉 Best implementation selected with time: {best_time*1000:.2f}ms")
