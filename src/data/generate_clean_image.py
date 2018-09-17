# coding : utf-8
import os, sys, time
import argparse
from PIL import Image
import glob, shutil
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Scrape line stamp')
    parser.add_argument('--base', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Base directory path to program files')
    parser.add_argument('--input_dir', type=str, default='../../data/raw',
                        help='Input directory')
    parser.add_argument('--output_dir', type=str, default='../../data/interim',
                        help='Output directory')
    args = parser.parse_args()

    input_dir = os.path.join(args.base, args.input_dir)
    result_dir = os.path.join(args.base, args.output_dir)

    # Stamp set loop
    box = []
    for stamp_set in range(360):
        stamp_dir = os.path.join(input_dir, '{0:04d}'.format(stamp_set))
        files = glob.glob('{}/*png'.format(stamp_dir))
        for f in files:
            img = Image.open(f)
            w, h = img.size
            r = 256 / float(min(w,h))
            img2 = img.resize((int(r*w), int(r*h)), Image.BICUBIC)
            plt.subplot(1,2,1)
            plt.imshow(img)
            plt.title('org')
            plt.subplot(1,2,2)
            plt.imshow(img2)
            plt.title('processed')
            plt.show()


if __name__ == '__main__':
    main()
