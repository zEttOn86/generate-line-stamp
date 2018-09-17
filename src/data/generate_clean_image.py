# coding : utf-8
import os, sys, time
import argparse
from PIL import Image
import glob

def main():
    parser = argparse.ArgumentParser(description='Scrape line stamp')
    parser.add_argument('--base', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Base directory path to program files')
    parser.add_argument('--input_dir', type=str, default='../../data/raw',
                        help='Input directory')
    parser.add_argument('--output_dir', type=str, default='../../data/interim',
                        help='Output directory')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size after resizing')
    args = parser.parse_args()

    num_of_imgs = 0
    # Stamp set loop
    for stamp_set in range(360):
        stamp_dir = os.path.join(args.base, args.input_dir, '{0:04d}'.format(stamp_set))
        result_dir = os.path.join(args.base, args.output_dir, '{0:04d}'.format(stamp_set))
        os.makedirs(result_dir, exist_ok=True)

        files = glob.glob('{}/*.png'.format(stamp_dir))
        for stamp_number, f in enumerate(files):
            img = Image.open(f)
            # Resize images so that (args.image_size, args.image_size)
            img = img.resize((args.image_size, args.image_size), Image.BICUBIC)
            img.save('{0:s}/{1:04d}.png'.format(result_dir, stamp_number))
            num_of_imgs += 1

        print('Finished stamp set: {} , # images: ~{}'.format(stamp_set, num_of_imgs))


if __name__ == '__main__':
    main()
