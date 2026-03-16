import torch
from matplotlib import pyplot as plt
from pathlib import Path

def denormalize(tensor, mean, std):
    """Denormalize a tensor given mean and std."""
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    return tensor * std + mean

def visualize_results(result, labels, data_mean, data_std, file):
    if result:
        # Denormalize to [0, 1] range for visualization
        orig_norm = denormalize(result.original_data, data_mean, data_std)
        recon_norm = denormalize(result.recreated_data, data_mean, data_std)
        orig_norm = torch.clamp(orig_norm, 0, 1)
        recon_norm = torch.clamp(recon_norm, 0, 1)
        
        print(f"\n✓ Attack completed!")
        print(f"  PSNR: {result.PSNR_score:.2f} dB")
        print(f"  SSIM: {result.SSIM_score:.4f}")
        
        # Save visualization
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = output_dir / file
        
        # Create visualization
        num_images = result.original_data.shape[0]
        fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))
        
        for i in range(num_images):
            # Original images
            orig_img = orig_norm[i].cpu().permute(1, 2, 0).numpy()
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f"Original\nLabel: {labels[i].item()}")
            axes[0, i].axis('off')
            
            # Reconstructed images
            recon_img = recon_norm[i].cpu().permute(1, 2, 0).numpy()
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title(f"Reconstructed")
            axes[1, i].axis('off')
        
        plt.suptitle(f"Geiping Attack (Oracle Labels) on CIFAR-10\nPSNR: {result.PSNR_score:.2f} dB | SSIM: {result.SSIM_score:.4f}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

def visualize_multiple_attacks(results, labels, data_mean, data_std, file):
    """Visualizes multiple attacks against the same client and data for comparison."""
    num_attacks = len(results)
    num_images = results[0].original_data.shape[0]
    
    # Denormalize all results
    denorm_results = []
    for result in results:
        orig_norm = denormalize(result.original_data, data_mean, data_std)
        recon_norm = denormalize(result.recreated_data, data_mean, data_std)
        orig_norm = torch.clamp(orig_norm, 0, 1)
        recon_norm = torch.clamp(recon_norm, 0, 1)
        denorm_results.append((orig_norm, recon_norm))
    
    # Save visualization
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / file
    
    # Create visualization with extra column for labels
    fig, axes = plt.subplots(num_attacks+1, num_images+1, figsize=(3*(num_images+1), 3*(num_attacks+1)))
    
    # Label column (leftmost)
    axes[0, 0].text(0.5, 0.5, "Original", ha='center', va='center', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    for j in range(num_attacks):
        label_text = f"Attack {j+1}\n\nPSNR: {results[j].PSNR_score:.2f} dB\nSSIM: {results[j].SSIM_score:.4f}"
        axes[j+1, 0].text(0.5, 0.5, label_text, ha='center', va='center', fontsize=12)
        axes[j+1, 0].axis('off')
    
    # Image columns
    for i in range(num_images):
        # Original images (only need to show once)
        orig_img = denorm_results[0][0][i].cpu().permute(1, 2, 0).numpy()
        axes[0, i+1].imshow(orig_img)
        axes[0, i+1].set_title(f"Label: {labels[i].item()}")
        axes[0, i+1].axis('off')
        
        for j in range(num_attacks):
            recon_img = denorm_results[j][1][i].cpu().permute(1, 2, 0).numpy()
            axes[j+1, i+1].imshow(recon_img)
            axes[j+1, i+1].axis('off')
    
    plt.suptitle(f"Comparison of Multiple Attacks on CIFAR-10")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")