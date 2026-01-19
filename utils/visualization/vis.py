from matplotlib import pyplot as plt

def plot_and_save(image, prompt, output_text, save_path):
    """Plot image with prompt and output text, then save as image file."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the image
    ax.imshow(image)
    ax.axis('off')
    
    # Add text box with prompt and output
    text_content = f"Prompt: {prompt}\n\nOutput: {output_text}"
    ax.text(0.02, 0.98, text_content, 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()