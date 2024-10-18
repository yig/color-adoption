import numpy as np
from PIL import Image
import argparse
from skimage import color
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def load_image(file_path):
    """Load an image and convert it to a numpy array."""
    return np.array(Image.open(file_path))

def rgb_to_lab(rgb_image):
    """Convert RGB image to LAB color space using skimage."""
    rgb_image = rgb_image.astype(float) / 255
    return color.rgb2lab(rgb_image)

def get_unique_colors(image_rgb, image_lab):
    """Get unique colors from RGB and LAB images."""
    flattened_rgb = image_rgb.reshape(-1, 3)
    flattened_lab = image_lab.reshape(-1, 3)
    
    # Get unique RGB colors and their indices
    _, unique_indices, inverse_indices = np.unique(flattened_rgb, axis=0, return_index=True, return_inverse=True)
    
    # Use the same indices to get corresponding LAB colors
    unique_rgb = flattened_rgb[unique_indices]
    unique_lab = flattened_lab[unique_indices]
    
    return unique_rgb, unique_lab, inverse_indices

def create_sparse_cost_matrix(colors1, colors2, k):
    """Create a sparse cost matrix based on K-nearest neighbors."""
    tree = cKDTree(colors2)
    distances, indices = tree.query(colors1, k=k)
    
    rows = np.repeat(np.arange(len(colors1)), k)
    cols = indices.ravel()
    data = distances.ravel()
    
    return csr_matrix((data, (rows, cols)), shape=(len(colors1), len(colors2)))

def match_colors(source_image, target_image, k=5):
    """Match colors between two images using sparse bipartite matching on unique colors."""
    # Convert images to LAB color space for matching
    source_lab = rgb_to_lab(source_image)
    target_lab = rgb_to_lab(target_image)
    
    # Get unique colors and their indices
    source_unique_rgb, source_unique_lab, source_inverse = get_unique_colors(source_image, source_lab)
    target_unique_rgb, target_unique_lab, _ = get_unique_colors(target_image, target_lab)
    
    # Create sparse cost matrix for K-nearest unique colors
    sparse_cost_matrix = create_sparse_cost_matrix(source_unique_lab, target_unique_lab, k)
    
    # Perform sparse bipartite matching
    matching = maximum_bipartite_matching(sparse_cost_matrix, perm_type='column')
    
    # Create color mapping in RGB space
    color_mapping_rgb = np.zeros_like(source_unique_rgb)
    valid_matches = matching != -1
    color_mapping_rgb[valid_matches] = target_unique_rgb[matching[valid_matches]]
    
    # Handle unmatched colors
    unmatched = ~valid_matches
    if np.any(unmatched):
        # For unmatched colors, find the nearest color in the target
        tree = cKDTree(target_unique_lab)
        _, nearest = tree.query(source_unique_lab[unmatched])
        color_mapping_rgb[unmatched] = target_unique_rgb[nearest]
    
    # Apply color mapping to the entire image in RGB space
    new_rgb = color_mapping_rgb[source_inverse].reshape(source_image.shape)
    
    return new_rgb.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Perform color transfer between two images using sparse bipartite matching.")
    parser.add_argument("source", help="Path to the source image")
    parser.add_argument("target", help="Path to the target image (color donor)")
    parser.add_argument("output", help="Path to save the output image")
    parser.add_argument("-k", type=int, default=5, help="Number of nearest neighbors to consider (default: 5)")
    args = parser.parse_args()

    source_image = load_image(args.source)
    target_image = load_image(args.target)
    
    result_image = match_colors(source_image, target_image, k=args.k)
    
    Image.fromarray(result_image).save(args.output)
    print(f"Color transfer completed. Result saved as '{args.output}'")

if __name__ == "__main__":
    main()
