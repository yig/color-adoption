import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import argparse

def load_image(file_path):
    """Load an image and convert it to a numpy array."""
    return np.array(Image.open(file_path))

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space."""
    r, g, b = rgb / 255.0
    r = ((r > 0.04045) * (((r + 0.055) / 1.055) ** 2.4) +
         (r <= 0.04045) * (r / 12.92)) * 100.0
    g = ((g > 0.04045) * (((g + 0.055) / 1.055) ** 2.4) +
         (g <= 0.04045) * (g / 12.92)) * 100.0
    b = ((b > 0.04045) * (((b + 0.055) / 1.055) ** 2.4) +
         (b <= 0.04045) * (b / 12.92)) * 100.0
    
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    x /= 95.047
    y /= 100.0
    z /= 108.883

    x = (x > 0.008856) * (x ** (1/3)) + (x <= 0.008856) * (7.787 * x + 16/116)
    y = (y > 0.008856) * (y ** (1/3)) + (y <= 0.008856) * (7.787 * y + 16/116)
    z = (z > 0.008856) * (z ** (1/3)) + (z <= 0.008856) * (7.787 * z + 16/116)

    return np.array([
        (116 * y) - 16,
        500 * (x - y),
        200 * (y - z)
    ])

def color_distance(c1, c2):
    """Calculate the Euclidean distance between two colors in LAB space."""
    return np.sqrt(np.sum((c1 - c2) ** 2))

def create_cost_matrix(colors1, colors2):
    """Create a cost matrix based on color distances."""
    return np.array([[color_distance(c1, c2) for c2 in colors2] for c1 in colors1])

def match_colors(source_image, target_image):
    """Match colors between two images using bipartite matching."""
    source_colors = source_image.reshape(-1, 3)
    target_colors = target_image.reshape(-1, 3)
    
    # Convert colors to LAB space
    source_lab = np.apply_along_axis(rgb_to_lab, 1, source_colors)
    target_lab = np.apply_along_axis(rgb_to_lab, 1, target_colors)
    
    # Create cost matrix
    cost_matrix = create_cost_matrix(source_lab, target_lab)
    
    # Perform bipartite matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create new image with matched colors
    new_colors = target_colors[col_ind]
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
