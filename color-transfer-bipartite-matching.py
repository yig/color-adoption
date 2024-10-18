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

def get_unique_colors(image):
    """Get unique colors from an image and their indices."""
    flattened = image.reshape(-1, 3)
    unique_colors, inverse_indices = np.unique(flattened, axis=0, return_inverse=True)
    return unique_colors, inverse_indices

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
    # Convert images to LAB color space
    source_lab = rgb_to_lab(source_image)
    target_lab = rgb_to_lab(target_image)
    
    # Get unique colors and their indices
    source_unique, source_inverse = get_unique_colors(source_lab)
    target_unique, target_inverse = get_unique_colors(target_lab)
    
    # Create sparse cost matrix for K-nearest unique colors
    sparse_cost_matrix = create_sparse_cost_matrix(source_unique, target_unique, k)
    
    # Perform sparse bipartite matching
    matching = maximum_bipartite_matching(sparse_cost_matrix, perm_type='column')
    
    # Create color mapping
    valid_matches = matching != -1
    color_mapping = np.zeros_like(source_unique)
    color_mapping[valid_matches] = target_unique[matching[valid_matches]]
    
    # Handle unmatched colors
    unmatched = ~valid_matches
    if np.any(unmatched):
        # For unmatched colors, find the nearest color in the target
        tree = cKDTree(target_unique)
        _, nearest = tree.query(source_unique[unmatched])
        color_mapping[unmatched] = target_unique[nearest]
    
    # Apply color mapping to the entire image
    new_lab = color_mapping[source_inverse].reshape(source_image.shape)
    
    # Convert back to RGB
    new_rgb = color.lab2rgb(new_lab) * 255
    
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
