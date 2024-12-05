import argparse

def main():
    parser = argparse.ArgumentParser(description="Perform color transfer between two images using sparse bipartite matching.")
    parser.add_argument("source", help="Path to the source image")
    parser.add_argument("reference", help="Path to the reference image (color donor)")
    parser.add_argument("output", help="Path to save the output image")
    parser.add_argument('--alg', type=str, default='GLO', 
                        choices=['GLO', 'BCC', 'PDF', 'CCS', 'MKL', 'GPC', 'FUZ', 'NST', 'DPT', 'TPS', 'HIS', 'PSN', 'EB3', 'CAM', 'RHG'],
                        help='Color transfer algorithm to use.')
    args = parser.parse_args()
    
    from ColorTransferLib.ColorTransfer import ColorTransfer
    from ColorTransferLib.ImageProcessing.Image import Image

    src = Image(file_path=args.source)
    ref = Image(file_path=args.reference)
    
    alg = args.alg
    ct = ColorTransfer(src, ref, alg)
    out = ct.apply()
    
    # No output file extension has to be given
    if out["status_code"] == 0:
        out["object"].write(args.output)
    else:
        print("Error: " + out["response"])
    
    print(f"Color transfer completed. Result saved as '{args.output}'")

if __name__ == "__main__":
    main()
