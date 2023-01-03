import torch 
import matplotlib.pyplot as plt
import numpy as np
import os


class TestEmbedding(torch.nn.Module):

    def __init__(self, vocab_size, embed_dim) -> None:
        super(TestEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc = torch.nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return torch.mean(x)

def b_to_gb(input):
    return input / 1000.0**3


def guess_embedding_size_in_gb(k, embed_dim, forward):
    vocab_size = 4**k // 2
    div_factor = 8 * (1000 ** 3)
    x = embed_dim * vocab_size * 32.0 / div_factor
    f = 1.0 if forward else 1.75
    return x * f


if __name__ == "__main__":
    
    dim = 128
    seq_len = 150
    batch_size = 2048

    device = torch.device("cuda:1")

    kmer_sizes = []
    inference_mems = []
    backward_mems = []
    
    for k in range(7, 13):
        vocab_size = 4**k // 2
        kmer_sizes.append(k)
        # inference
        model = TestEmbedding(vocab_size, dim)
        model = model.to(device)
        with torch.no_grad():
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            x = x.to(device)
            out = model(x)
            mem_1 = b_to_gb(torch.cuda.memory_reserved(device))
            inference_mems.append(mem_1)

        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = y.to(device)
        out_2 = model(y)
        scalar_out = torch.sum(out_2)
        scalar_out.backward()
        mem_2 = b_to_gb(torch.cuda.memory_reserved(device))
        backward_mems.append(mem_2)

        del model

    for k in range(13, 15):
        kmer_sizes.append(k)
        inference_mems.append(guess_embedding_size_in_gb(k, dim, True))
        backward_mems.append(guess_embedding_size_in_gb(k, dim, False))
    
    print(kmer_sizes)
    print(inference_mems)
    print(backward_mems)

    fig, axes = plt.subplots()
    
    bar_width = 0.25

    x1 = np.arange(0, len(kmer_sizes))
    x2 = [x + bar_width for x in x1]
    
    axes.bar(x1, inference_mems, color="red", width=bar_width, edgecolor="white", label="Inference memory")
    axes.bar(x2, backward_mems, color="blue", width=bar_width, edgecolor="white", label="Training memory")

    axes.set_xlabel("k-mer size")
    axes.set_ylabel("Memory consumption in GB")
    axes.set_xticks([x + bar_width / 2.0 for x in range(len(kmer_sizes))])
    axes.set_xticklabels(kmer_sizes)

    axes.legend()
    axes.set_title(f"Memory consumption of an embedding layer (Dim=128)")
    
    fig.tight_layout()
    fig.savefig(os.path.join("../../result_images", "embedding_mem_consumption.pdf"), transparent=False)
        