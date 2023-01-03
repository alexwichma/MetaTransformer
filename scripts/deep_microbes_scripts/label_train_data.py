import argparse
import glob
import os
from progress.bar import IncrementalBar


def write_class_mappings(label_path, species_arr, genus_arr, multi_level):
    species_fname = "species_mapping.tab" if not multi_level else "species_mapping_multi_level.tab"
    genus_fname = "genus_mapping.tab" if not multi_level else "genus_mapping_multi_level.tab"
    species_out_file = os.path.join(label_path, species_fname)
    genus_out_file = os.path.join(label_path,genus_fname)
    with open(species_out_file, 'w') as file_handle:
        file_handle.write("Class\tSpecies\n")
        for index, species_name in enumerate(species_arr):
            file_handle.write(str(index) + "\t" + species_name + "\n")
    with open(genus_out_file, 'w') as file_handle:
        file_handle.write("Class\tGenus\n")
        for index, genus_name in enumerate(genus_arr):
            file_handle.write(str(index) + "\t" + genus_name + "\n")


def parse_classes(label_path, multi_level):
    genus_set = set()
    species_arr = []
    file_2_genus = {}

    files = glob.glob(os.path.join(label_path, "taxonomy_*.tab"))
    for f in files:
        with open(f, 'r') as file_handle:
            lines = file_handle.readlines()[1:]
            for line in lines:
                line_split = line.rstrip("\n").split("\t")
                file_name = line_split[0]
                # sometimes there are rows with incomplete data (e.g. only two ranks filled)
                genus_name = line_split[-1] if len(line_split) == 6 or len(line_split) == 7 else "NA"
                # only add species and genus to classes if genus exists for certain species (multi-level usage)
                if multi_level and genus_name == "NA":
                    continue
                species_arr.append(file_name)
                if genus_name != "NA":
                    genus_set.add(genus_name)
                    file_2_genus[file_name] = genus_name

    return sorted(species_arr), sorted(genus_set), file_2_genus


# create path if not exists
def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


# label a sequence file with taxId on genus level (if present)
def label_train_genus(src_file_path, out_path, species_mapping, genus_mapping, species_2_genus_mapping):
    file_name = os.path.basename(src_file_path)
    file_name_no_ending = file_name.rstrip(".fa")
    # if there is no appropriate genus available for this species, skip genus labeling
    if not file_name_no_ending in species_2_genus_mapping:
        return
    species_name = file_name_no_ending
    species_class_idx = species_mapping.index(species_name)
    genus_name = species_2_genus_mapping[species_name]
    genus_class_index = genus_mapping.index(genus_name)
    # collect same genus sequences in the same file (grouped by genus name), this is needed for ART simulator later on
    out_file_path = os.path.join(out_path, f'{genus_name}.fa')
    with open(src_file_path, 'r') as src_file_handle:
        with open(out_file_path, 'a') as out_file_handle:
            for src_file_line in src_file_handle:
                if src_file_line.startswith(">"):
                    line = f">lbl|{genus_class_index}|{genus_name}|{species_class_idx}|{species_name}\n"
                    out_file_handle.write(line)
                else:
                    out_file_handle.write(src_file_line)


# label a sequence file with taxId on species level (here, filename)
def label_train_species(src_file_path, out_path, sorted_species, genus_mapping, species_2_genus_mapping, multi_level):
    with open(src_file_path, 'r') as src_file_handle:
        file_name = os.path.basename(src_file_path)
        file_name_no_ending = file_name.rstrip(".fa")
        species_name = file_name_no_ending
        # species class is missing a corresponding genus level mapping -> skip it 
        if multi_level and species_name not in sorted_species:
            return
        species_class_index = sorted_species.index(species_name)
        genus_name = species_2_genus_mapping[species_name] if species_name in species_2_genus_mapping else "NA"
        genus_idx = genus_mapping.index(genus_name) if genus_name != "NA" else -1
        out_file_path = os.path.join(out_path, file_name)
        with open(out_file_path, 'w') as out_file_handle:
            for src_file_line in src_file_handle:
                if src_file_line.startswith(">"):
                    line = f">lbl|{species_class_index}|{species_name}|{genus_idx}|{genus_name}\n"
                    out_file_handle.write(line)
                else:
                    out_file_handle.write(src_file_line)


# iterates over all available training sequences and labels them on species and genus level
def label_train_data(train_path, species_mapping, genus_mapping, species_2_genus_mapping, multi_level):
    src_file_paths = glob.glob(os.path.join(train_path, "*.fa"))
    bar = IncrementalBar("Progress", max=len(src_file_paths))
    
    if not multi_level:
        species_train_path = os.path.join(train_path, "..", "labeled_species_sequences")
        genus_train_path = os.path.join(train_path, "..", "labeled_genus_sequences")
        create_path(species_train_path)
        create_path(genus_train_path)
        for src_file_path in src_file_paths:
            label_train_species(src_file_path, species_train_path, species_mapping, genus_mapping, species_2_genus_mapping, False)
            label_train_genus(src_file_path, genus_train_path, species_mapping, genus_mapping, species_2_genus_mapping)
            bar.next()
    else:
        multi_level_train_path = os.path.join(train_path, "..", "labeled_multi_level")
        create_path(multi_level_train_path)
        for src_file_path in src_file_paths:
            label_train_species(src_file_path, multi_level_train_path, species_mapping, genus_mapping, species_2_genus_mapping, True)
            bar.next()


def main():
    parser = argparse.ArgumentParser(
        description="Labels deep microbes training data and creates class-index <-> taxonomy mappings.")
    parser.add_argument("--train-path", dest="train_path", type=str, help="Path to directory containing raw genomes", required=True)
    parser.add_argument("--tax-path", dest="tax_path", type=str, help="Path to directory containing genus and species taxonomy files", required=True)
    parser.add_argument("--multi-level", dest="multi_level", action="store_true", help="Whether to label data on both levels simutaenously", default=False)
    args = parser.parse_args()

    train_path = args.train_path
    tax_path = args.tax_path
    multi_level = args.multi_level

    print("Loading label definitions...")
    species_class_mapping, genus_class_mapping, species_2_genus_mapping = parse_classes(tax_path, multi_level)
    print("Creating mappings from label definitions...")
    write_class_mappings(tax_path, species_class_mapping, genus_class_mapping, multi_level)
    print("Labeling training data on species and genus level...")
    label_train_data(train_path, species_class_mapping, genus_class_mapping, species_2_genus_mapping, multi_level)


if __name__ == "__main__":
    main()