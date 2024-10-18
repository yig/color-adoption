import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import argparse
from skimage import color

def load_image(file_path):
    """Load an image and convert it to a numpy array."""
    return np.array(Image.open(file_path))

def rgb_to_lab(rgb_image):
    """Convert RGB image to LAB color space using skimage."""
    # Ensure the image is in float format and in the range [0, 1]
    rgb_image = rgb_image.astype(float) / 255
    return color.rgb2lab(rgb_image)

def color_distance(c1, c2):
    """Calculate the Euclidean distance between two colors in LAB space."""
    return np.sqrt(np.sum((c1 - c2) ** 2))

def create_cost_matrix(colors1, colors2):
    """Create a cost matrix based on color distances."""
    return np.array([[color_distance(c1, c2) for c2 in colors2] for c1 in colors1])

def match_colors(source_image, target_image):
    """Match colors between two images using bipartite matching."""
    # Convert images to LAB color space
    source_lab = rgb_to_lab(source_image)
    target_lab = rgb_to_lab(target_image)
    
    # Reshape to 2D arrays of colors
    source_colors = source_lab.reshape(-1, 3)
    target_colors = target_lab.reshape(-1, 3)
    
    # Create cost matrix
    cost_matrix = create_cost_matrix(source_colors, target_colors)
    
    # Perform bipartite matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create new image with matched colors
    new_colors = target_image.reshape(-1, 3)[col_ind]
    new_image = new_colors.reshape(source_image.shape)
    
    return new_image.astype(np.uint8)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Perform color transfer between two images using bipartite matching.")
    parser.add_argument("source", help="Path to the source image")
    parser.add_argument("target", help="Path to the target image (color donor)")
    parser.add_argument("output", help="Path to save the output image")
    args = parser.parse_args()

    # Load source and target images
    source_image = load_image(args.source)
    target_image = load_image(args.target)
    
    # Match colors
    result_image = match_colors(source_image, target_image)
    
    # Save the result
    Image.fromarray(result_image).save(args.output)
    print(f"Color transfer completed. Result saved as '{args.output}'")

if __name__ == "__main__":
    main()
