from __future__ import absolute_import, division, print_function
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit
from argparse import ArgumentParser

# use N instead of <unk> here since otherwise this leads to problem when the tokens are split into single characters
SPECIAL_TOKENS = ["<pad>", "N", "<cls>"]


if __name__ == "__main__":
    parser = ArgumentParser(description="Generates a byte-pair-encoding for the given input tokens")
    parser.add_argument("--in-path", dest="in_path", type=str, help="Path to tokenized file", required=True)
    parser.add_argument("--out-path", dest="out_path", type=str, help="Path where BpeModel is saved", required=True)

    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path

    tokenizer = Tokenizer(BPE(unk_token="N"))
    trainer = BpeTrainer(vocab_size=2**22, special_tokens=SPECIAL_TOKENS)
    tokenizer.pre_tokenizer = CharDelimiterSplit("\n")
    tokenizer.train([in_path], trainer)
    tokenizer.save(out_path)