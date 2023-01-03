import argparse
from logging import root
import os

def write_fasta(reads, counter, rootname, out):

        if (counter < 10):
            pre = "00"
        elif (counter < 100):
            pre = "0"
        else:
            pre = ""

        outname = "split" + pre + str(counter)+ "_" + rootname 
        outname = os.path.join(out, outname)
        
        o = open (outname, "w")
        for i in reads: o.write(i)
        o.close()

def main():

    parser = argparse.ArgumentParser(description="Reads fasta file and splits into multiple files with n reads")
    parser.add_argument("--input", dest="fname", type=str, required=True)
    parser.add_argument("--n-reads", dest="n", type=int, required=False , default=40000)
    parser.add_argument("--out-folder", dest="out", type=str, required=True)

    args = parser.parse_args()
    fname = args.fname
    out = args.out
    n = args.n
    
    
    f = open (fname, "r")
    
    rootname = os.path.basename(fname)
    reads = list()
    counter = 0
    name_counter = 0

    for i in f:
          
        if i.startswith(">"):

            if (counter >= n):

                write_fasta(reads, name_counter, rootname, out)
                name_counter += 1
                reads.clear()
                counter = 0
            
            counter += 1
        
        reads.append(i)
    
    write_fasta(reads, name_counter, rootname, out)


if __name__ == "__main__":
    main()
