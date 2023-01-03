from __future__ import absolute_import, division, print_function
from tokenizers import Tokenizer
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", dest="model_path", type=str, help="Path to BpeModel", required=True)
    parser.add_argument("--test-file-path", dest="test_file_path", type=str, help="Path to test reads", required=True)

    args = parser.parse_args()
    model_path = args.model_path
    test_file_path = args.test_file_path

    model = Tokenizer.from_file(model_path)
    print(model.get_vocab_size())
    
    sum = 0
    count = 0
    with open(test_file_path, "r") as file_handle:
        file_handle.readline()
        buffer = ""
        for line in file_handle:
            if line.startswith(">"):
                ids = model.encode(buffer).ids
                sum += len(ids)
                count += 1
                buffer = ""
                print(f"Mean fragment length {sum/count}", end="\r")
            else:
                line = line.strip()
                buffer += line

    