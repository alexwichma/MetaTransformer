"""
Testing script that calculates and saves:
    - read-level precision and recall (across classification thresholds from 0 to 1 in 0.05 steps)
    - community-level precision and recall (across abundance cut offs 0,0.01,0.0001)
    - normalized abundance estimations as well as absolute error to the ground-truth abundance per class 
    - relative and normalized abundance of multiple-single or multiple-splited files
"""


from __future__ import absolute_import, division, print_function




import copy
import torch
from tqdm import tqdm
import os
import glob
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


def test(net, test_loader, device, index, single = True):

    out_predictions = []
    out_labels = []
    if single: file_ids = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            if single:
                reads, labels, file_id = data
            else:
                reads, labels = data

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
            
            out_predictions.append(out_copy)
            out_labels.append(labels_copy)

            if single:
                file_id_copy = torch.tensor(file_id)
                file_ids.append(file_id_copy)

            
    out_predictions = torch.cat(out_predictions)
    out_labels = torch.cat(out_labels).long()
    if single: 
        file_ids = torch.cat(file_ids).long()
        return out_predictions, out_labels, file_ids
    else:
        return out_predictions, out_labels

def evaluate_predictions(predictions, labels, class_mapping, out_path, threshold):
    results_abundances_path = os.path.join(out_path, "abundances.csv")
    results_commm_lv_path = os.path.join(out_path, "comm_lv_results.csv")
    results_read_lv_path = os.path.join(out_path, "read_lv_results.csv")

    read_lv_cols = ["threshold", "precision", "recall", "f1-score"]
    thresholds = np.array([0.05 * x for x in range(0, 21)], dtype=np.float)
    read_lv_results = []
    
    # Test thresholds from 0% to 100% for read-level precision, use default (.5) for community level
    for t in thresholds:
        read_prec, read_rec, read_f1 = read_level_metrics(predictions, labels, t)
        read_lv_results.append([t, read_prec, read_rec, read_f1])

    comm_lv_cols = ["cutoff", "precision", "recall"]
    cut_offs = np.array([0.0, 0.01, 0.0001], dtype=np.float)
    comm_lv_results = []

    class_names = list(class_mapping.values())
    num_classes = len(class_names)
    abundance_cols = ["Taxon", "Prediction", "GroundTruth", "Error"]
    abundance_results = None

    for cut_off in cut_offs:
        comm_prec, comm_rec, comm_abundances = community_level_metrics(predictions, labels, num_classes,threshold, cut_off)
        comm_lv_results.append([cut_off, comm_prec, comm_rec])
        # write out abundance estimation only for community_level results without cutoff (cutoff 0.0)
        if cut_off == 0.0:
            abundance_results = []

            for class_name, prediction, ground_truth, error in zip(class_names, *comm_abundances):
                abundance_results.append([class_name, prediction, ground_truth, error])
    
    logger.info(abundance_cols, abundance_results)
    
    # create dataframes
    read_lv_df = pd.DataFrame(read_lv_results, columns=read_lv_cols)
    comm_lv_df = pd.DataFrame(comm_lv_results, columns=comm_lv_cols)
    abundance_df = pd.DataFrame(abundance_results, columns=abundance_cols)

    # round and write out csv
    read_lv_df = round_dataframe(read_lv_df)
    comm_lv_df = round_dataframe(comm_lv_df)
    abundance_df = round_dataframe(abundance_df)
    read_lv_df.to_csv(results_read_lv_path)
    comm_lv_df.to_csv(results_commm_lv_path)
    abundance_df.to_csv(results_abundances_path)

def calculate_abundance(predictions, labels, class_mapping, out_path, threshold, fname):
    outname = fname + "_abundance.csv" 
    results_abundances_path = os.path.join(out_path, outname)
    
    class_names = list(class_mapping.values())
    num_classes = len(class_names)
    abundance_cols = ["Taxon", "Prediction", "Norm_Prediction"]
    abundance_results = []
    
    abundances, norm_abundances =   get_abundance(predictions, num_classes, threshold)
   
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
    
    #print("number of files")
    #print(len(ids))
    
    for i in ids:
        name = map_ids.get(i)
        name = name.split("/")
        name = name[len(name)-1]
        name = name[:-3]
        name = str(fname) + "_" + name
        filter_id = torch.nonzero(torch.where(file_ids == i, 1, 0))

        p = predictions[filter_id]
        l = labels[filter_id]

        p = torch.squeeze(p)
        l = torch.squeeze(l)
        

        calculate_abundance(p, l, class_mapping, out_path, threshold, name)


def main():
    fix_random_seeds()
    cmd_args = parse_args_test()
    
    single = cmd_args.single
    if (single == "True"): single = True 
    else: single = False  
    #print ("Single mode: " + str(single))

    labeled = cmd_args.labeled
    if (labeled == "True" ): labeled = True
    else: labeled = False
    #print("Labeled mode: " + str(labeled) )
    
    multiple = cmd_args.multiple
    if (multiple == "True" ): multiple = True
    else: multiple = False


    init_device_handler(False, 1, 0, False)
   
    hydra_conf= OmegaConf.load(cmd_args.config_state_path)
    
    vocab, vocab_size = None, 0
    if not hydra_conf.mdl_common.input_module in ["lsh", "hash", "one_hot", "one_hot_embed", "bpe"]:
        vocab, vocab_size = load_vocabulary(cmd_args.vocab_path)
    else:
        print("Vocabulary is not loaded since {} is used".format(hydra_conf.mdl_common.input_module))
    num_classes = hydra_conf.mdl_common.num_classes

    out_path = cmd_args.output_path

    class_mapping = parse_class_mapping(cmd_args.class_mapping_path)

    device = device_handler.get_device()

    read_transforms = read_transforms_for_input_layer(hydra_conf.mdl_common.input_module, hydra_conf, vocab, train=False)
    label_transform = get_label_transforms(hydra_conf.mdl_common.class_indices, labeled=labeled)
    net = instantiate_model_by_str_name(hydra_conf.model.name, hydra_conf, vocab_size)

    state = torch.load(cmd_args.model_state_path, map_location="cpu")
    net.load_state_dict(state["model_state_dict"], strict=False)
    net.eval()
    net.to(device)
    

    if multiple == True:
      all_folder = glob.glob(os.path.join(cmd_args.reads_path , "*/"))
      for folders in all_folder:
        all_files = glob.glob(os.path.join(folders, "*.fa"))
        file_map = dict()
        file_map_rev = dict()
    
        root_name = os.path.basename(folders[:-1])
        print(root_name)
        print(folders)
        
        if single:
          for i in range(len(all_files)):
            file_map[all_files[i]] = i
            file_map_rev[i] = all_files[i]     
        if single:
            dataset = ProcessingMetagenomicSingleReadDataset(folders, read_transforms, file_map, label_transforms=label_transform, consistent_read_len=False)
            dataloader = DataLoader(dataset, batch_size=cmd_args.batch_size, shuffle=False, num_workers=cmd_args.n_worker, collate_fn=train_collate_single_fn_padded, drop_last=True)
        else:
            dataset = ProcessingMetagenomicReadDataset(folders, read_transforms, label_transforms=label_transform, consistent_read_len=False)
            dataloader = DataLoader(dataset, batch_size=cmd_args.batch_size, shuffle=False, num_workers= cmd_args.n_worker, collate_fn=train_collate_fn_padded, drop_last=True)
    
        avg_n = cmd_args.prediction_mode
        threshold = hydra_conf.mdl_common.classification_threshold
    
       
        if single:
           predictions, labels , file_ids = test(net, dataloader, device, cmd_args.level_index, single)
           file_ids = file_ids[torch.arange(0, file_ids.shape[0], avg_n)]
        else:
            predictions, labels = test(net, dataloader, device, cmd_args.level_index, single)
        
        # average the predictions
        predictions = average_n_predictions(predictions, num_classes, avg_n)
        labels = labels[torch.arange(0, labels.shape[0], avg_n)]
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        assert predictions.shape[0] == labels.shape[0]
        
        if single:
            calculate_single_abundance(predictions, labels, file_ids, class_mapping, out_path, file_map_rev, threshold, cmd_args.fname)
        else:
            calculate_abundance(predictions, labels, class_mapping, out_path, threshold, cmd_args.fname + root_name)
        
        del predictions
        del labels 
        del dataset
        del dataloader
        if single: del file_ids

    else:    
      all_files = glob.glob(os.path.join(cmd_args.reads_path, "*.fa"))
      file_map = dict()
      file_map_rev = dict()
      
      root_name = os.path.basename(cmd_args.reads_path)
      if single:
        for i in range(len(all_files)):
            file_map[all_files[i]] = i
            file_map_rev[i] = all_files[i]
      
      #print(file_map)
      
      
      if single:
          dataset = ProcessingMetagenomicSingleReadDataset(cmd_args.reads_path, read_transforms, file_map, label_transforms=label_transform, consistent_read_len=False)
          dataloader = DataLoader(dataset, batch_size=cmd_args.batch_size, shuffle=False, num_workers=cmd_args.n_worker, collate_fn=train_collate_single_fn_padded, drop_last=True)
      else:
          dataset = ProcessingMetagenomicReadDataset(cmd_args.reads_path, read_transforms, label_transforms=label_transform, consistent_read_len=False)
          dataloader = DataLoader(dataset, batch_size=cmd_args.batch_size, shuffle=False, num_workers= cmd_args.n_worker, collate_fn=train_collate_fn_padded, drop_last=True)
  
      avg_n = cmd_args.prediction_mode
      threshold = hydra_conf.mdl_common.classification_threshold
  
     
      if single:
         predictions, labels , file_ids = test(net, dataloader, device, cmd_args.level_index, single)
         file_ids = file_ids[torch.arange(0, file_ids.shape[0], avg_n)]
      else:
          predictions, labels = test(net, dataloader, device, cmd_args.level_index, single)
      
      # average the predictions
      predictions = average_n_predictions(predictions, num_classes, avg_n)
      labels = labels[torch.arange(0, labels.shape[0], avg_n)]
      predictions = torch.nn.functional.softmax(predictions, dim=1)
      assert predictions.shape[0] == labels.shape[0]
      
      if single:
          calculate_single_abundance(predictions, labels, file_ids, class_mapping, out_path, file_map_rev, threshold, cmd_args.fname)
      else:
          calculate_abundance(predictions, labels, class_mapping, out_path, threshold, cmd_args.fname + root_name)
      
    #evaluate_predictions(predictions, labels, class_mapping, out_path, threshold)


if __name__ == "__main__":
    main()
