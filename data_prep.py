#!/usr/bin/env python3
import argparse, os, random
from PIL import Image

def resize_and_save(in_path, out_path, size):
    img = Image.open(in_path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    img.save(out_path)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train','val','test']:
        for t in ['raw','edited']:
            os.makedirs(os.path.join(args.output_dir, split, t), exist_ok=True)

    # gather paired filenames
    raws = sorted(os.listdir(args.raw_dir))
    edits = sorted(os.listdir(args.edited_dir))
    pairs = list(zip(raws, edits))
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * args.train_frac)
    n_val   = int(n * args.val_frac)
    # rest goes to test
    splits = (
        ('train', pairs[:n_train]),
        ('val',   pairs[n_train:n_train+n_val]),
        ('test',  pairs[n_train+n_val:])
    )

    for split_name, items in splits:
        for raw_fn, edit_fn in items:
            resize_and_save(
                os.path.join(args.raw_dir, raw_fn),
                os.path.join(args.output_dir, split_name, 'raw', raw_fn),
                args.size
            )
            resize_and_save(
                os.path.join(args.edited_dir, edit_fn),
                os.path.join(args.output_dir, split_name, 'edited', edit_fn),
                args.size
            )

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir',    type=str, required=True)
    p.add_argument('--edited_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--size',       type=int, default=224,
                   help='resize resolution (square)')
    p.add_argument('--train_frac', type=float, default=0.9)
    p.add_argument('--val_frac',   type=float, default=0.05)
    args = p.parse_args()
    main(args)
