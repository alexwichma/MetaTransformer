import time

from torch.utils.data.dataloader import DataLoader
from dataset.MetagenomicReadDataset import ProcessingMetagenomicReadDataset
from utils.transforms import Read2VocabKmer, TrimRead
from utils.utils import load_vocabulary


if __name__ == "__main__":
    vocab, vocab_size = load_vocabulary("../data/shared/vocabs/vocab_12mer.txt")
    transforms = [TrimRead(), Read2VocabKmer(vocab, 12, 150)]
    ds = ProcessingMetagenomicReadDataset("../data/hgr_umgs/test_raw/test_mock/mock_labeled/mock_metagenome_01.1.fa.fa", vocab, 12, transforms)
    batch_size = 4096
    dl = DataLoader(ds, batch_size)

    start_time = time.time()
    items_processed = 0
    for item in dl:
        items_processed += batch_size
        time_elapsed = time.time() - start_time
        print("Throughput: {} elements/second".format(items_processed / time_elapsed))

    print("Total time: {} Total elements: {}".format(time.time() - start_time, items_processed))