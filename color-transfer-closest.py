import numpy as np
from PIL import Image
import argparse
from skimage import color
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

def load_image(file_path):
    """Load an image and convert it to a numpy array."""
    return np.array(Image.open(file_path).convert('RGB'))

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

def match_colors(source_image, target_image):
    """Match colors between two images using sparse bipartite matching on unique colors."""
    # Convert images to LAB color space for matching
    source_lab = rgb_to_lab(source_image)
    target_lab = rgb_to_lab(target_image)
    
    # Get unique colors and their indices
    source_unique_rgb, source_unique_lab, source_inverse = get_unique_colors(source_image, source_lab)
    target_unique_rgb, target_unique_lab, target_inverse = get_unique_colors(target_image, target_lab)

    print( f"Source image has {len(source_unique_rgb)} unique colors." )
    print( f"Target image has {len(target_unique_rgb)} unique colors." )

    if len( target_unique_rgb ) > len( source_unique_rgb ):
        print("WARNING: There are more unique colors in the target (donor) image than in the source image. Not every donor color will be used.")
    
    # Create color mapping in RGB space by finding the closest color (in LAB space).
    tree = cKDTree(target_unique_lab)
    _, nearest = tree.query(source_unique_lab)
    updated_source_unique_rgb = target_unique_rgb[nearest]
    
    # Apply color mapping to the entire image in RGB space
    new_rgb = updated_source_unique_rgb[source_inverse].reshape(source_image.shape)
    
    return new_rgb.astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Perform color transfer between two images using sparse bipartite matching.")
    parser.add_argument("source", help="Path to the source image")
    parser.add_argument("target", help="Path to the target image (color donor)")
    parser.add_argument("output", help="Path to save the output image")
    args = parser.parse_args()

    source_image = load_image(args.source)
    target_image = load_image(args.target)
    
    result_image = match_colors(source_image, target_image)
    
    Image.fromarray(result_image).save(args.output)
    print(f"Color transfer completed. Result saved as '{args.output}'")

if __name__ == "__main__":
    main()
