"""
Testing script that calculates and saves:
    - read-level precision and recall (across classification thresholds from 0 to 1 in 0.05 steps)
    - community-level precision and recall (across abundance cut offs 0,0.01,0.0001)
    - normalized abundance estimations as well as absolute error to the ground-truth abundance per class 
"""


from __future__ import absolute_import, division, print_function

# temporary
import glob
#


import copy
import torch
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.MetagenomicReadDataset import ProcessingMetagenomicReadDataset, ProcessingMetagenomicSingleReadDataset
from models.model_utils import get_label_transforms, instantiate_model_by_str_name, read_transforms_for_input_layer
from logging import getLogger
from utils import parse_class_mapping, round_dataframe
from utils import device_handler
from utils.device_handler import init_device_handler
from utils.metric_utils import average_n_predictions, community_level_metrics, read_level_metrics, get_abundance
from utils.argument_parser import parse_args_test
from utils.torch_utils import train_collate_fn_padded, train_collate_single_fn_padded
from utils.utils import fix_random_seeds, load_vocabulary


logger = getLogger('test/main')


def test(net, test_loader, device, index):

    out_predictions = []
    out_labels = []
    file_ids = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            reads, labels, file_id = data

            if device.type == "cuda":
                reads = reads.to(device)

            with torch.cuda.amp.autocast(enabled=True):
                out = net(reads)
                if index != -1:
                    out = out[index]
            
            # save predictions to prediction tensor for later analaysis
            # deepcopy the tensors to avoid tons of open FDs due to multiprocessing
            # see: https://pytorch.org/docs/stable/multiprocessing.html#file-descriptor-file-descriptor
            # i can not observe the same behaviour in other testing scenarios, which is a bit odd
            out = out.float()
            out = out.cpu()
            out_copy = copy.deepcopy(out)
            labels_copy = copy.deepcopy(labels)
            file_id_copy = torch.tensor(file_id)
            
            out_predictions.append(out_copy)
            out_labels.append(labels_copy)
            file_ids.append(file_id_copy)
    
                  
    out_predictions = torch.cat(out_predictions)
    out_labels = torch.cat(out_labels).long()
    file_ids = torch.cat(file_ids).long()  
    
    return out_predictions, out_labels, file_ids

def merge(data, labels, file_ids):
    
   
    ids = list()
    uni = list()

    for i in range(len(file_ids)):
        tmp = np.asarray(file_ids[i])
        tmp = np.unique(tmp)
        if (len(tmp)==1):
            uni.append(True)
        else:
            uni.append(False)
        ids.extend(tmp)

    ids = np.unique(ids)
    
    #print(uni)
    #print(ids)

    merged_label = list() 
    merged_data = list()
    merged_ids = list()

    print("test \n" + str(data[0][1])+ " "  + str(data[0].shape() ))
    for i in range(len(ids)):

        tmp_label = list()
        tmp_data = list()
        tmp_ids = list()

        for j in range(len(file_ids)):
            if (uni[j] and ids[i] == file_ids[j][0]):
                tmp_label.extend(labels[j])
                tmp_data.extend(data[j])
                tmp_ids.extend(file_ids[j])
            else:
                for k in range(len(file_ids[j])):
                    if (ids[i] == file_ids[j][k]):
                        tmp_label.append(labels[j][k])
                        tmp_data.append(data[j][k])
                        tmp_ids.append(file_ids[j][k])
        
        merged_label.append(torch.cat(tmp_label))
        merged_data.append(torch.cat(tmp_data))
        merged_ids.append(tmp_ids)

    print(len(merged_data))        
    

    for i in merged_data:
        print(i)
        #print(len(i))
    
    return  merged_data, merged_label, merged_ids, ids


def calculate_abundance(predictions, labels, class_mapping, out_path, threshold, fname):
    
    results_abundances_path = os.path.join(out_path, fname)
    results_commm_lv_path = os.path.join(out_path, "comm_lv_results.csv")
    results_read_lv_path = os.path.join(out_path, "read_lv_results.csv")

    class_names = list(class_mapping.values())
    num_classes = len(class_names)
    abundance_cols = ["Taxon", "Prediction", "Norm_Prediction"]
    abundance_results = []
    
    abundances, norm_abundances = get_abundance(predictions, num_classes, threshold)
   
    for class_name, prediction, Norm_Prediction in zip(class_names, abundances, norm_abundances):
        abundance_results.append([class_name, prediction, Norm_Prediction])
    
    logger.info(abundance_cols, abundance_results)
    
    # create dataframes
    abundance_df = pd.DataFrame(abundance_results, columns=abundance_cols)

    # round and write out csv
    abundance_df = round_dataframe(abundance_df)
    abundance_df.to_csv(results_abundances_path)

def calculate_single_abundance(predictions, labels, file_ids, class_mapping, out_path, map_ids, threshold, fname = ""):
    
    ids = map_ids.keys()
    
    for i in ids:
        name = map_ids.get(i)
        name = name.split("/")
        name = name[len(name)-1]
        name = name[:-3]
        name = str(fname) + "_" + name + "_abundance.csv"
        filter_id = torch.nonzero(torch.where(file_ids == i, 1, 0))
        
        p = predictions[filter_id]
        l = labels[filter_id]
        
        p = torch.squeeze(p)
        l = torch.squeeze(l)
        
        calculate_abundance(p, l, class_mapping, out_path, threshold, name)


def main():
    fix_random_seeds()
    cmd_args = parse_args_test()

    init_device_handler(False, 1, 0, False)
   
    hydra_conf = OmegaConf.load(cmd_args.config_state_path)
    
    vocab, vocab_size = None, 0
    if not hydra_conf.mdl_common.input_module in ["lsh", "hash", "one_hot", "one_hot_embed", "bpe"]:
        vocab, vocab_size = load_vocabulary(cmd_args.vocab_path)
    else:
        print("Vocabulary is not loaded since {} is used".format(hydra_conf.mdl_common.input_module))
    num_classes = hydra_conf.mdl_common.num_classes

    out_path = cmd_args.output_path

    class_mapping = parse_class_mapping(cmd_args.class_mapping_path)

    device = device_handler.get_device()
    
    #test_tmp (cmd_args.reads_path)
    
    
    read_transforms = read_transforms_for_input_layer(hydra_conf.mdl_common.input_module, hydra_conf, vocab, train=False)
    label_transform = get_label_transforms(hydra_conf.mdl_common.class_indices)
    net = instantiate_model_by_str_name(hydra_conf.model.name, hydra_conf, vocab_size)

    state = torch.load(cmd_args.model_state_path, map_location="cpu")
    net.load_state_dict(state["model_state_dict"], strict=False)
    net.eval()
    net.to(device)
    
    all_files = glob.glob(os.path.join(cmd_args.reads_path, "*.fa"))
    file_map = dict()
    file_map_rev = dict()
    
    for i in range(len(all_files)): 
        file_map[all_files[i]] = i 
        file_map_rev[i] = all_files[i] 
    
    dataset = ProcessingMetagenomicSingleReadDataset(cmd_args.reads_path, read_transforms, file_map, label_transforms=label_transform, consistent_read_len=False)
    dataloader = DataLoader(dataset, batch_size=cmd_args.batch_size, shuffle=False, num_workers=64, collate_fn=train_collate_single_fn_padded, drop_last=True)
    
    avg_n = cmd_args.prediction_mode
    threshold = hydra_conf.mdl_common.classification_threshold

    predictions, labels , file_ids = test(net, dataloader, device, cmd_args.level_index)
    
    print(file_ids.shape)
    print(labels.shape)
    print(predictions.shape)
    
    predictions = average_n_predictions(predictions, num_classes, avg_n)
    labels = labels[torch.arange(0, labels.shape[0], avg_n)]
    file_ids = file_ids[torch.arange(0, file_ids.shape[0], avg_n)]
    
    predictions = torch.nn.functional.softmax(predictions, dim=1)
    assert predictions.shape[0] == labels.shape[0]
    
    f = file_ids[torch.nonzero(torch.where(file_ids == 1, 1, 0))]
    l = labels[torch.nonzero(torch.where(file_ids == 1, 1, 0))]
    p = predictions[torch.nonzero(torch.where(file_ids == 1, 1, 0))]
    print(file_map_rev.get(1))
    
    print(f.shape)
    print(l.shape)
    print(p.shape)
    
    f = torch.squeeze(f)
    p = torch.squeeze(p)
    
    print(f.shape)
    print(p.shape)
    print(p)
    
    f = file_ids[torch.nonzero(torch.where(file_ids == 3, 1, 0))]
    l = labels[torch.nonzero(torch.where(file_ids == 3, 1, 0))]
    p = predictions[torch.nonzero(torch.where(file_ids == 3, 1, 0))]
    print(file_map_rev.get(3))
    
    print(f.shape)
    print(l.shape)
    print(p.shape)
    
    f = torch.squeeze(f)
    p = torch.squeeze(p)
    
    print(f.shape)
    print(p.shape)
    print(p)
    
    print(file_ids.shape)
    print(labels.shape)
    print(predictions.shape)
    
    
    calculate_single_abundance(predictions, labels, file_ids, class_mapping, out_path, file_map_rev, threshold, "test")
        
    #print(predictions)
    
    #predictions, labels, file_ids, unique_ids = merge (predictions, labels, file_ids )
    
    #print(len(file_ids))
    #print(len(labels))
    #print(len(predictions))
    #print(unique_ids)
       
    #print (len(file_ids[0]))
    #print (labels[0].shape[0])

    
    
if __name__ == "__main__":
    main()
