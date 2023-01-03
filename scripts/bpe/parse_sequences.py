from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from glob import glob
import os
from tqdm import tqdm
from random import shuffle


unknown_token = "<unk>"
translation_dict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
                    "K": "N", "M": "N", "R": "N", "Y": "N", "S": "N",
                    "W": "N", "B": "N", "V": "N", "H": "N", "D": "N",
                    "X": "N"}
OVERLAP = 20
CANONICAL = False


def forward2reverse(seq):
    letters = list(seq)
    letters = [translation_dict[base] for base in letters]
    return ''.join(letters)[::-1]


def make_canonical(seq):
    seq_r = forward2reverse(seq)
    return seq if seq < seq_r else seq_r


def parse_files(file_paths, out_file, token_len):
    lines = []
    for file_path in tqdm(file_paths):
        with open(file_path, "r") as file_handle:
            # skip first header line
            file_handle.readline()
            buffer = ""
            for line in file_handle:
                if not line.startswith(">"):
                    line = line.strip()
                    line = line.upper()
                    buffer += line
                if line.startswith(">"):
                    lines.append(buffer)
                    buffer = ""
    
    tokens = []
    for l in tqdm(lines):
        for i in range(0, len(l), token_len - OVERLAP):
            token = l[i:(i + token_len)]
            token = make_canonical(token) if CANONICAL else token
            tokens.append(token)
    
    shuffle(tokens)

    with open(out_file, "w") as file_handle:
        for token in tqdm(tokens):
            file_handle.write(token)
            file_handle.write("\n")  


if __name__ == "__main__":
    parser = ArgumentParser(description="Parses raw genome data as input for byte-pair-encoding training")
    parser.add_argument("--in-dir", dest="in_dir", type=str, help="Folder path to genome data", required=True)
    parser.add_argument("--out-path", dest="out_path", type=str, help="Path to output file", required=True)
    parser.add_argument("--token-len", dest="token_len", type=int, help="Lengths of the tokens to create", default=150)
    args = parser.parse_args()

    in_dir = args.in_dir
    out_path = args.out_path
    token_len = args.token_len

    paths = glob(os.path.join(in_dir, "*.fa"))
    parse_files(paths, out_path, token_len)