import numpy as np
from PIL import Image
import argparse
from skimage import color
from scipy.spatial import cKDTree
import ot

def load_image(file_path):
    """Load an image and convert it to a numpy array."""
    return np.array(Image.open(file_path).convert('RGB'))

def rgb_to_lab(rgb_image):
    """Convert RGB image to LAB color space using skimage."""
    rgb_image = rgb_image.astype(float) / 255
    return color.rgb2lab(rgb_image)

def get_unique_colors(image_rgb, image_lab):
    """Get unique colors from RGB and LAB images with their frequencies."""
    flattened_rgb = image_rgb.reshape(-1, 3)
    flattened_lab = image_lab.reshape(-1, 3)
    
    unique_rgb, inverse_indices, counts = np.unique(flattened_rgb, axis=0, return_inverse=True, return_counts=True)
    
    unique_lab = flattened_lab[np.unique(inverse_indices)]
    frequencies = counts / counts.sum()
    
    return unique_rgb, unique_lab, inverse_indices, frequencies

def wasserstein_color_transfer(source_image, target_image):
    """Transfer colors using Wasserstein distance."""
    source_lab = rgb_to_lab(source_image)
    target_lab = rgb_to_lab(target_image)
    
    source_unique_rgb, source_unique_lab, source_inverse, source_freq = get_unique_colors(source_image, source_lab)
    target_unique_rgb, target_unique_lab, _, target_freq = get_unique_colors(target_image, target_lab)
    
    # Compute cost matrix
    cost_matrix = ot.dist(source_unique_lab, target_unique_lab)
    
    # Normalize frequencies
    source_freq_norm = source_freq / source_freq.sum()
    target_freq_norm = target_freq / target_freq.sum()
    
    # Compute optimal transport
    transport_plan = ot.emd(source_freq_norm, target_freq_norm, cost_matrix)
    
    # Map colors based on transport plan
    mapped_colors = np.dot(transport_plan, target_unique_rgb)
    
    # Apply color mapping to the entire image in RGB space
    new_rgb = mapped_colors[source_inverse].reshape(source_image.shape)
    
    return new_rgb.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Perform color transfer between two images using Wasserstein distance.")
    parser.add_argument("source", help="Path to the source image")
    parser.add_argument("target", help="Path to the target image (color donor)")
    parser.add_argument("output", help="Path to save the output image")
    args = parser.parse_args()

    source_image = load_image(args.source)
    target_image = load_image(args.target)
    
    result_image = wasserstein_color_transfer(source_image, target_image)
    
    Image.fromarray(result_image).save(args.output)
    print(f"Color transfer completed. Result saved as '{args.output}'")

if __name__ == "__main__":
    main()
