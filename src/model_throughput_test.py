from __future__ import absolute_import, division, print_function
from omegaconf.omegaconf import OmegaConf
import torch
from torch.utils.data.dataloader import DataLoader
from dataset.MetagenomicReadDataset import ProcessingMetagenomicReadDataset
from models.model_utils import get_label_transforms, instantiate_model_by_str_name, read_transforms_for_input_layer

from utils import device_handler
from utils.torch_utils import train_collate_fn_padded
from utils.utils import load_vocabulary
from tqdm import tqdm
import time


device_handler.init_device_handler(False, 1, 0, False)

device = device_handler.get_device()

path = "/share/ebuschon/model_tests/mem_tf_test_2021-07-24_08:36/config.yaml"
vocab_path = "/share/ebuschon/data/shared/vocabs/vocab_12mer.txt"
reads = "/share/ebuschon/data/hgr_umgs/reads/test_mock_bi_interleaved/mock01_split"

hydra_conf = OmegaConf.load(path)

vocab, vocab_size = None, 0
if not hydra_conf.mdl_common.input_module in ["lsh", "hash", "one_hot", "one_hot_embed", "bpe"]:
    vocab, vocab_size = load_vocabulary(vocab_path)


num_classes = hydra_conf.mdl_common.num_classes

# load model
read_transforms = read_transforms_for_input_layer(hydra_conf.mdl_common.input_module, hydra_conf, vocab, train=False)
label_transform = get_label_transforms(hydra_conf.mdl_common.class_indices)
net = instantiate_model_by_str_name(hydra_conf.model.name, hydra_conf, vocab_size)

net.to(device)

prefetch_factor = 2
batch_size_multiply = 1
batch_size = 2048 * batch_size_multiply
amp = True


dataset = ProcessingMetagenomicReadDataset(reads, read_transforms, label_transforms=label_transform, consistent_read_len=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=64, collate_fn=train_collate_fn_padded, prefetch_factor=prefetch_factor)

total = 10000000
count = 0
start = time.time()
with torch.no_grad():
    for data, _ in tqdm(dataloader, total=total//batch_size):
        data = data.to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            out = net(data)
        count += batch_size
        if count > total:
            break
end = time.time()

time_in_minutes = (end - start) / 60.0
rpm = count / time_in_minutes

print(rpm)
