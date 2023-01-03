import glob
import os
from tqdm import tqdm


in_folder = "/share/ebuschon/data/hgr_umgs/reads/test_gut_species"
out_folder ="/share/ebuschon/data/hgr_umgs/reads/test_gut_species_multi_level"
species_mapping_path = "/share/ebuschon/data/hgr_umgs/train_raw/sequence_metadata/species_mapping.tab"
multi_lv_species_mapping_path = "/share/ebuschon/data/hgr_umgs/train_raw/sequence_metadata/species_mapping_multi_level.tab"

index_to_species = {}
multi_level_species_to_index = {}

with open(species_mapping_path, "r") as f:
    f.readline()
    for line in f:
        line = line.strip()
        split = line.split("\t")
        class_idx = int(split[0])
        name = split[1]
        index_to_species[class_idx] = name

with open(multi_lv_species_mapping_path, "r") as f:
    f.readline()
    for line in f:
        line = line.strip()
        split = line.split("\t")
        class_idx = int(split[0])
        name = split[1]
        multi_level_species_to_index[name] = class_idx


in_paths = glob.glob(os.path.join(in_folder, "*.fa"))

for in_path in tqdm(in_paths):
    basename = os.path.basename(in_path)
    out_path = os.path.join(out_folder, basename)
    with open(in_path, "r") as in_handle:
        with open(out_path, "w") as out_handle:
            write_line = True
            for line in in_handle:
                if line.startswith(">"):
                    line_split = line.split("|")
                    class_id = int(line_split[1])
                    species_name = index_to_species[class_id]
                    if species_name in multi_level_species_to_index:
                        line_split[1] = str(multi_level_species_to_index[species_name])
                        write_line = True
                        out_handle.write("|".join(line_split))
                    else:
                        write_line = False
                if write_line and not line.startswith(">"):
                    out_handle.write(line)


