from __future__ import absolute_import, division, print_function

import argparse

# arguments for the testing scripts
def parse_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads-path", dest="reads_path", type=str, help="Path to input reads directory", required=True)
    parser.add_argument("--output-path", dest="output_path", type=str, help="Path where output should be saved", required=True)
    parser.add_argument("--class-mapping-path", dest="class_mapping_path",  type=str, help="Path to class mapping", required=True)
    parser.add_argument("--vocab-path", dest="vocab_path",  type=str, help="Path to k-mer vocabulary", required=True)
    parser.add_argument("--model-state", dest="model_state_path", type=str, help="Path to .pt checkpoint file", required=True)
    parser.add_argument("--config-state", dest="config_state_path", type=str, help="Path to config.yaml checkpoint file", required=True)
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size used", default=2048)
    parser.add_argument("--prediction-mode", dest="prediction_mode", type=int, help="Indicates whether single(1), paired(2) or paired+reverse-complement(4) prediction is performed", required=True)
    parser.add_argument("--level-index", dest="level_index", type=int, help="Level to use for multi-level models", default=-1)
    parser.add_argument("--n-worker", dest="n_worker", type=int, help="Number of usable CPU cores for processing", default=80)
    parser.add_argument("--filename", dest="fname", type=str, help="Name of abundance file", default="")
    parser.add_argument("--single-files", dest="single",  help="Set True if one file per estimation is used, otherwise set False", default="True", type=str)
    parser.add_argument("--labeled", dest="labeled", help="Set True if dataset is labeled via | seperator; default:False", default="False", type=str) 
    parser.add_argument("--multiple-folder" , dest="multiple", help="Set True if files are split into multiple folder in inout folder, otherwise set False")
    config = parser.parse_args()


    return config
