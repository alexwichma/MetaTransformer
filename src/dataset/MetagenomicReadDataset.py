from __future__ import absolute_import, division, print_function

from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.transforms import Compose
import torch
import os
import glob
import random
import numpy as np
from threading import Thread

import logging

from tqdm import tqdm
from utils.utils import load_vocabulary

# @DEPRECATED
class DatasetChunkIterator:

    def __init__(self, dataset_path, repeat_on_exhaust=True, context_name='', prepend_cls_token=False):
        self.dataset_path = dataset_path
        self.prepend_cls_token = prepend_cls_token
        self.read_files = []
        self.label_files = []
        self.is_first_run = True
        self.repeat_on_exhaust = repeat_on_exhaust
        self.logger = logging.getLogger(DatasetChunkIterator.__name__ + '/' + context_name)
        self.populate_files()
        # Buffer to hold dataset pre-loaded in separate thread
        self.buffer_ds = None
        self.buffer_thread: Thread = self.construct_thread()
        self.buffer_thread.start()

    # retrieve all dataset chunks from given dir, shuffle them randomly
    def populate_files(self):
        read_paths = os.path.join(self.dataset_path, "reads_*.npy")
        label_paths = os.path.join(self.dataset_path, "labels_*.npy")
        reads = sorted(glob.glob(read_paths))
        labels = sorted(glob.glob(label_paths))
        zipped = list(zip(reads, labels))
        random.shuffle(zipped)
        read_files, label_files = zip(*zipped)
        self.read_files = list(read_files)
        self.label_files = list(label_files)

    def __iter__(self):
        return self

    def construct_thread(self):
        return Thread(group=None, target=self.fetch_next_ds, daemon=True)

    # method meant to be run in a thread to prefetch next ds chunk
    # since this is mainly IO bound, it should result in a speedup
    def fetch_next_ds(self):
        read_file = self.read_files.pop()
        label_file = self.label_files.pop()
        self.logger.debug("Next read file path: {}".format(read_file))
        self.logger.debug("Next label file path: {}".format(label_file))
        self.buffer_ds = ProcessedMetagenomicReadDataset(read_file, label_file, prepend_cls_token=self.prepend_cls_token)

    def __next__(self):
        # handle empty dataset chunk queue
        if len(self.read_files) == 0 and len(self.label_files) == 0:
            if not self.is_first_run and not self.repeat_on_exhaust:
                raise StopIteration()
            elif not self.is_first_run and self.repeat_on_exhaust:
                self.logger.info("All input files were processed ones. Starting over.")
            else:
                self.is_first_run = False
            self.populate_files()
        
        if self.buffer_thread.is_alive():
            self.logger.debug("No buffered dataset available yet. Waiting for thread to finish loading.")
            self.buffer_thread.join()
        else:
            self.logger.debug("Using buffered dataset for next call.")
        
        ds_to_return = self.buffer_ds
        self.buffer_ds = None
        self.buffer_thread = self.construct_thread()
        self.buffer_thread.start()
        return ds_to_return


# @DEPRECATED Plain metagenomic read dataset operating on preprocessed npy files
class ProcessedMetagenomicReadDataset(Dataset):

    def __init__(self, read_file_path, label_file_path):
        self.reads = np.load(read_file_path)
        self.labels = np.load(label_file_path)

    def __len__(self):
        return self.reads.shape[0]

    def __getitem__(self, idx):
        read = self.reads[idx]
        label = self.labels[idx]
        return torch.tensor(read, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class ProcessingMetagenomicReadDataset(IterableDataset):
    """
    Custom IterableDataset implementation that reads raw read data from files and applies read and 
    label transforms before returning an element. It supports multiple workers.
    """

    def __init__(self, file_path, read_transforms, directory_mode=True, label_transforms=None, consistent_read_len=True) -> None:
        super().__init__()
        self.logger = logging.getLogger(ProcessingMetagenomicReadDataset.__name__)
        self.file_path = file_path
        if directory_mode:
            self.all_files = glob.glob(os.path.join(self.file_path, "*.fa"))
        else:
            self.all_files = [file_path]
        self.worker_files = None
        self.current_file_handle = None
        self.read_label_buffer = None
        self.transforms = Compose(read_transforms)
        self.label_transform = label_transforms
        self.lbl_extract_fn = self.label_transform if self.label_transform is not None else self.extract_label_from_header
        self.read_len_cache = None
        self.consistent_read_len = consistent_read_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or len(self.all_files) == 1:
            self.logger.debug("Using dataset with a single worker")
            self.worker_files = self.all_files
        else:
            num_workers = worker_info.num_workers
            assert len(self.all_files) / num_workers >= 1 , f"Number files({len(self.all_files)}) smaller than {num_workers} worker. All files {self.all_files}"
            wk_id = worker_info.id
            self.logger.debug("Initializing worker {}/{}".format(wk_id + 1, num_workers))
            fpw, res = divmod(len(self.all_files), num_workers)
            self.worker_files = self.all_files[wk_id * fpw + min(wk_id, res):(wk_id + 1) * fpw + min(wk_id + 1, res)]
            random.shuffle(self.worker_files)
        return self

    def skip_to_first_header(self, f_handle):
        current: str = f_handle.readline()
        while not current.startswith(">"):
            current = f_handle.readline()
        return current

    def __next__(self):
        if self.current_file_handle is None:
            if len(self.worker_files) > 0:
                self.current_file_handle = open(self.worker_files.pop(0), "r")
                first_header_line = self.skip_to_first_header(self.current_file_handle)
                self.read_label_buffer = self.lbl_extract_fn(first_header_line)
            else:
                raise StopIteration()
        read_lines = []
        label = self.read_label_buffer
        while True:
            line = self.current_file_handle.readline().strip()
            if line.startswith(">"):
                self.read_label_buffer = self.lbl_extract_fn(line)
                read = "".join(read_lines)
                # buffer length of first complete read, this is used to filter out incomplete reads at the of a file
                if self.read_len_cache is None:
                    self.read_len_cache = len(read)
                read_tensor = torch.tensor(self.transforms(read), dtype=torch.long)
                label_tensor = label
                return read_tensor, label_tensor
            elif line :
                read_lines.append(line)
            else:
                self.current_file_handle.close()
                self.current_file_handle = None
                # empty read found at end of file skip to next file, just skip to next file
                if len(read_lines) == 0:
                    return self.__next__()
                read = "".join(read_lines)
                # read smaller than normal read length found, skip directly to next file
                if self.consistent_read_len and len(read) < self.read_len_cache:
                    return self.__next__()
                read_tensor = torch.tensor(self.transforms(read), dtype=torch.long)
                return read_tensor, label

    # fallback label transform function in-case no label transform is specified  
    def extract_label_from_header(self, header):
        return int(header.split("|")[1])



class ProcessingMetagenomicSingleReadDataset(IterableDataset):
    """
    Custom IterableDataset implementation that reads raw read data from multiple single files and applies read and 
    label transforms before returning an element. It supports multiple workers.
    """

    def __init__(self, file_path, read_transforms, file_map, directory_mode=True, label_transforms=None, consistent_read_len=True) -> None:
        super().__init__()
        self.logger = logging.getLogger(ProcessingMetagenomicReadDataset.__name__)
        self.file_path = file_path
        if directory_mode:
            self.all_files = glob.glob(os.path.join(self.file_path, "*.fa"))
        else:
            self.all_files = [file_path]
        self.worker_files = None
        self.current_file_handle = None
        self.read_label_buffer = None
        self.transforms = Compose(read_transforms)
        self.label_transform = label_transforms
        self.lbl_extract_fn = self.label_transform if self.label_transform is not None else self.extract_label_from_header
        self.read_len_cache = None
        self.consistent_read_len = consistent_read_len
        self.file_map = file_map

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or len(self.all_files) == 1:
            self.logger.debug("Using dataset with a single worker")
            self.worker_files = self.all_files
        else:
            num_workers = worker_info.num_workers
            assert len(self.all_files) / num_workers >= 1 , f"Number files({len(self.all_files)}) smaller than {num_workers} worker. All files {self.all_files}"
            wk_id = worker_info.id
            self.logger.debug("Initializing worker {}/{}".format(wk_id + 1, num_workers))
            fpw, res = divmod(len(self.all_files), num_workers)
            self.worker_files = self.all_files[wk_id * fpw + min(wk_id, res):(wk_id + 1) * fpw + min(wk_id + 1, res)]
        return self

    def skip_to_first_header(self, f_handle):
        current: str = f_handle.readline()
        while not current.startswith(">"):
            current = f_handle.readline()
        return current
        
    def __next__(self):
        if self.current_file_handle is None:
            if len(self.worker_files) > 0:

                self.current_file_handle = open(self.worker_files.pop(0), "r")
                first_header_line = self.skip_to_first_header(self.current_file_handle)
                self.read_label_buffer = self.lbl_extract_fn(first_header_line)
            else:
                raise StopIteration()
        read_lines = []
        label = self.read_label_buffer
        
        curr_file = self.current_file_handle.name
        f_id = 1
        file_m = dict(self.file_map)
        f_id = file_m[curr_file]
        #f_id = torch.tensor(f_id)
        
        while True:
            line = self.current_file_handle.readline().strip()
            if line.startswith(">"):
                self.read_label_buffer = self.lbl_extract_fn(line)
                read = "".join(read_lines)
                # buffer length of first complete read, this is used to filter out incomplete reads at the of a file
                if self.read_len_cache is None:
                    self.read_len_cache = len(read)
                read_tensor = torch.tensor(self.transforms(read), dtype=torch.long)
                label_tensor = label
                return read_tensor, label_tensor, f_id
            elif line :
                read_lines.append(line)
            else:
                self.current_file_handle.close()
                self.current_file_handle = None
                # empty read found at end of file skip to next file, just skip to next file
                if len(read_lines) == 0:
                    return self.__next__()
                read = "".join(read_lines)
                # read smaller than normal read length found, skip directly to next file
                if self.consistent_read_len and len(read) < self.read_len_cache:
                    return self.__next__()
                read_tensor = torch.tensor(self.transforms(read), dtype=torch.long)
                return read_tensor, label, f_id

    # fallback label transform function in-case no label transform is specified  
    def extract_label_from_header(self, header):
        return int(header.split("|")[1])



