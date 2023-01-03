cimport cython
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t, int32_t
from libc.string cimport strchr, strcmp, strlen, strcpy, strncpy
import numpy as np
cimport numpy as cnp
import xxhash


cdef extern from "MurmurHash3.cpp":
    void MurmurHash3_x86_32 (const void *key, int len, uint32_t seed, void *out) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
cdef kmer_lsh(char* kmer_str, int l, int b, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] seeds, uint32_t num_seeds, uint32_t num_lmers):
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sketch = np.zeros((num_seeds,), dtype=np.int32)
    cdef int32_t [:] sketch_mem_view = sketch
    cdef int32_t MAX_VAL = 2147483647 # 2 * 31 - 1
    cdef int32_t hash_val[1]
    cdef int32_t min_hash_val
    cdef uint32_t s, i
    # iterate over each hash function, for each hash function calculate all l-mer hashes and save minumum one
    for s in range(num_seeds):
        min_hash_val = MAX_VAL
        for i in range(num_lmers):
            MurmurHash3_x86_32(kmer_str, l, seeds[s], hash_val)
            if hash_val[0] < min_hash_val:
                min_hash_val = hash_val[0]
            kmer_str += 1
        sketch_mem_view[s] = min_hash_val
        # kmer_str is a char pointer therefore we used pointer arithmetics to reset to beginning of k-mer
        kmer_str -= l
    return sketch


@cython.boundscheck(False)
@cython.wraparound(False)
cdef char* rev_compl(char* seq, uint32_t seq_len):
    # calculates reverse complement of a sequence
    cdef char* rev_compl_seq = <char*> malloc(sizeof(char) * seq_len)
    cdef uint32_t i, idx
    for i in range(seq_len):
        idx = seq_len - 1 - i
        if seq[idx] == b"G":
            rev_compl_seq[i] = b"C"
        elif seq[idx] == b"C":
            rev_compl_seq[i] = b"G"
        elif seq[idx] == b"T":
            rev_compl_seq[i] = b"A"
        else:
            rev_compl_seq[i] = b"T"
    return rev_compl_seq


@cython.boundscheck(False)
@cython.wraparound(False)
def seq_to_kmer_lsh(char* seq, int k, int l, int b, cnp.ndarray[dtype=cnp.uint32_t, ndim=1] seeds not None, int unk_idx, int offset):
    cdef uint32_t seq_len = strlen(seq)
    cdef uint32_t num_kmers = seq_len - k + 1
    cdef uint32_t num_lmers = k - l + 1
    cdef uint32_t half_num_lmers = num_lmers // 2 + 1
    cdef uint32_t num_seeds = len(seeds)
    cdef uint32_t i
    cdef cnp.ndarray[cnp.int32_t, ndim=1] data = np.empty((num_kmers,), dtype=np.int32)
    cdef int32_t [:] data_mem_view = data
    # pointers for the k-mer, its reverse complement and unknown position query
    cdef char* kmer_f = <char*> malloc(sizeof(char) * k)
    cdef char* kmer_r
    cdef char* unkPos
    for i in range(num_kmers):
        strncpy(kmer_f, seq, k)
        # check if the k-mer contains an unkwown element, if so just set the position to unknown index and continue
        unkPos = strchr(kmer_f, b"N")
        if unkPos:
            data_mem_view[i] = unk_idx
            seq += 1
            unkPos = NULL
            continue
        # calc canonical k-mer
        kmer_r = rev_compl(kmer_f, k)
        kmer_c = kmer_f if strcmp(kmer_f, kmer_r) >= 0 else kmer_r
        res = kmer_lsh(kmer_c, l, b, seeds, num_seeds, num_lmers)
        res.sort()
        half_res = res[:half_num_lmers]
        signature_bytes = half_res.tobytes()
        data_mem_view[i] = (xxhash.xxh32(signature_bytes).intdigest() & (b - 1)) + offset 
        # free memory for reverse_complement to avoid memory leak
        free(kmer_r)
        # go to next k-mer
        seq += 1
    # free memory for forward k-mer memory to avoid memory leak
    free(kmer_f)
    return data


