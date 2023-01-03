"""
Utility around taxonomic trees and ranks. The methods are not really used in the rest of the implementation
since i sticked to ground_truth information in the read headers.
"""


from __future__ import absolute_import, division, print_function

from enum import Enum
from typing import List


class TaxonomicRank(Enum):
    GENOME = "Genome"
    SUBSPECIES = "Subspecies"
    SPECIES = "Species"
    GENUS = "Genus"
    FAMILY = "Family"
    ORDER = "Order"
    CLASS = "Class"
    PHYLUM = "Phylum"
    KINGDOM = "Kingdom"
    DOMAIN = "Domain"
    ROOT = "ROOT"

def str_to_rank(rank: str) -> TaxonomicRank:
    opts = {
        "Genome": TaxonomicRank.GENOME,
        "Subspecies": TaxonomicRank.SUBSPECIES,
        "Species": TaxonomicRank.SPECIES,
        "Genus": TaxonomicRank.GENUS,
        "Family": TaxonomicRank.FAMILY,
        "Order": TaxonomicRank.ORDER,
        "Class": TaxonomicRank.CLASS,
        "Phylum": TaxonomicRank.PHYLUM,
        "Kingdom": TaxonomicRank.KINGDOM,
        "Domain": TaxonomicRank.DOMAIN,
    }
    return opts[rank]


child_to_parent = {
            TaxonomicRank.GENOME: TaxonomicRank.SPECIES,
            TaxonomicRank.SUBSPECIES: TaxonomicRank.SPECIES,
            TaxonomicRank.SPECIES: TaxonomicRank.GENUS,
            TaxonomicRank.GENUS: TaxonomicRank.FAMILY,
            TaxonomicRank.FAMILY: TaxonomicRank.ORDER,
            TaxonomicRank.ORDER: TaxonomicRank.CLASS,
            TaxonomicRank.CLASS: TaxonomicRank.PHYLUM,
            TaxonomicRank.PHYLUM: TaxonomicRank.KINGDOM,
            TaxonomicRank.KINGDOM: TaxonomicRank.DOMAIN
        }

parent_to_child = {v: k for k, v in child_to_parent.items()}


class TaxonomicNode:

    def __init__(self, name, unique_id, class_id, rank):
        self.name = name
        self.unique_id = unique_id
        self.class_id = class_id
        self.rank = rank
        self.parent = None
        self.children = []

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent
    
    def get_rank(self):
        return self.rank

    def get_class_id(self):
        return self.class_id

    def get_unique_id(self):
        return self.unique_id

    def get_tax_name(self):
        return self.name

    def get_children(self):
        return self.children

    def add_child(self, child):
        self.children.append(child)


class CustomTaxonomicTree:

    def __init__(self, taxonomy_path, class_mapping_path, undetermined_token="NA", undetermined_class_id=-1) -> None:
        self.taxonomy_path = taxonomy_path
        self.class_mapping_path = class_mapping_path

        # helper for now to match existing header format in data
        self.genome_class_id_to_unique_id = {}
        self.unique_id_to_node = {}
        self.rank_to_num_classes = {}
        # cache to hold lineages full lineages for faster access e.g. statistics
        self.complete_line_age_class_id_cache = {}

        self.undetermined_token = undetermined_token
        self.undetermined_class_id = undetermined_class_id
        self.ranks = None
        self.rank_to_index = None
        self.parse_class_mapping()
        self.parse_taxonomy()

    def get_parent_rank(self, rank: TaxonomicRank) -> TaxonomicRank:
        return child_to_parent[rank]

    def get_child_rank(self, rank: TaxonomicRank) -> TaxonomicRank:
        return parent_to_child[rank]

    def genome_class_id_to_tax_name(self, class_id):
        uid = self.genome_class_id_to_unique_id[class_id]
        node = self.unique_id_to_node[uid]
        return node.get_tax_name()

    def nodes_to_class_ids(self, nodes):
        result = []
        for node in nodes:
            result.append(node.get_class_id() if node is not None else self.undetermined_class_id)
        return result

    def get_node_lineage(self, node: TaxonomicNode, selected_ranks, return_class_ids=False):
        result = []
        current_node: TaxonomicNode = node
        while current_node is not None:
            if current_node.get_rank() in selected_ranks:
                result.append(current_node)
            current_node = current_node.get_parent()
        result_len = len(result)
        selecetion_len = len(selected_ranks)
        # if there are missing ranks extend by undetermined class_id since parent is unknown
        result.extend([None] * (selecetion_len - result_len))

        if return_class_ids:
            return self.nodes_to_class_ids(result)

        return result

    def lineage_by_genome_class_id(self, class_id,  selected_ranks=[TaxonomicRank.GENOME, TaxonomicRank.GENUS], return_class_ids=False):
        uid = self.genome_class_id_to_unique_id[class_id]
        current_node: TaxonomicNode = self.unique_id_to_node[uid]
        return self.get_node_lineage(current_node, selected_ranks, return_class_ids=return_class_ids)

    def lineage_by_unique_id(self, unique_id, selected_ranks=[TaxonomicRank.GENOME, TaxonomicRank.GENUS], return_class_ids=False):
        return self.get_node_lineage(self.unique_id_to_node[unique_id], selected_ranks, return_class_ids=return_class_ids)

    def gen_cls_id_to_parent_cls_id(self, class_id, rank):
        if rank not in self.ranks:
            return self.undetermined_class_id
        uid = self.genome_class_id_to_unique_id[class_id]
        current_node: TaxonomicNode = self.unique_id_to_node[uid]
        while current_node is not None and current_node.get_rank() != rank:
            current_node = current_node.get_parent()
        if current_node is None or current_node.get_rank() != rank:
            return self.undetermined_class_id
        return current_node.get_class_id()
       
    def get_available_ranks(self):
        return self.ranks

    def get_num_classes_for_rank(self, rank):
        return self.rank_to_num_classes[rank]

    def get_rank_order_permutation(self, ranks):
        permutation = []
        current_rank = TaxonomicRank.GENOME
        while len(permutation) != len(ranks):
            if current_rank in ranks:
                permutation.append(ranks.index(current_rank))
            current_rank = self.get_parent_rank(current_rank)
        return permutation

    def parse_taxonomy(self):
        with open(self.taxonomy_path, "r") as file:
            # get available ranks from header and order them ascending
            ranks = [str_to_rank(rank) for rank in file.readline().strip().split("\t")]
            permutation = self.get_rank_order_permutation(ranks) 
            self.ranks = [ranks[index] for index in permutation]
            self.rank_to_index = {rank: index for index, rank in enumerate(self.ranks)}
            
            for line in file:
                tax_names = line.strip().split("\t")
                # sometimes some columns in the file are left empty if lineage is unknown
                if len(tax_names) < len(ranks):
                    tax_names.extend([self.undetermined_token] * (len(ranks) - len(tax_names)))
                sorted_tax_names = [tax_names[index] for index in permutation]
                # iterate over child, parent pairs and add them to each other as child and parent
                for child_taxon_uid, parent_taxon_uid in zip(sorted_tax_names[:-1], sorted_tax_names[1:]):
                    if child_taxon_uid != self.undetermined_token and parent_taxon_uid != self.undetermined_token:
                        child_node = self.unique_id_to_node[child_taxon_uid]
                        parent_node = self.unique_id_to_node[parent_taxon_uid]
                        child_node.set_parent(parent_node)
                        parent_node.add_child(child_node)

    def parse_class_mapping(self):
        current_rank = None
        class_count = None
        with open(self.class_mapping_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    if class_count is not None:
                        self.rank_to_num_classes[current_rank] = class_count
                    class_count = 0
                    current_rank = str_to_rank(line[1:-1])
                else:
                    split = line.split("\t")
                    class_id = int(split[0])
                    tax_name = split[1]
                    # for now the tax_name is the unique id, since this is how the data is layout
                    unique_id = tax_name
                    node = TaxonomicNode(tax_name, unique_id, class_id, current_rank)
                    self.unique_id_to_node[unique_id] = node

                    if current_rank == TaxonomicRank.GENOME:
                        self.genome_class_id_to_unique_id[class_id] = unique_id

                    class_count += 1
            # catch last rank after last line parsed
            self.rank_to_num_classes[current_rank] = class_count
                    

if __name__ == "__main__":
    ctt = CustomTaxonomicTree("../data/hgr_umgs/train_raw/sequence_metadata_tax_tools/taxonomy.tab","../data/hgr_umgs/train_raw/sequence_metadata_tax_tools/full_class_mapping.tab")
    
    ranks = [TaxonomicRank.GENOME, TaxonomicRank.GENUS, TaxonomicRank.FAMILY]
    all_ranks = ctt.get_available_ranks()

    # test all the methods
    print(ctt.get_rank_order_permutation(ranks))
    print("\n")
    for rank in ranks:
        print("Rank {} has {} classes".format(rank, ctt.get_num_classes_for_rank(rank)))
    print("\n")
    print(ctt.get_available_ranks())
    print("\n")

    test_unique_id = "12718_7_39"
    test_genome_class_id = 16

    # expected results Firmicutes   Clostridia  Clostridiales   Peptostreptococcaceae   Clostridium (rtl <-)
    node_lineage_by_unique_id: List[TaxonomicNode] = ctt.lineage_by_unique_id(test_unique_id, all_ranks)
    for node in node_lineage_by_unique_id:
        print("Node name: {} | Node rank: {} | Node class_id: {}".format(node.get_tax_name(), node.get_rank(), node.get_class_id()))

    print("\n")
    node_lineage_by_genome_id = ctt.lineage_by_genome_class_id(test_genome_class_id, all_ranks)
    for node in node_lineage_by_genome_id:
        print("Node name: {} | Node rank: {} | Node class_id: {}".format(node.get_tax_name(), node.get_rank(), node.get_class_id()))

    print("\n")
    print(ctt.genome_class_id_to_unique_id[test_genome_class_id])

    print("\n")
    for rank in all_ranks:
        print(ctt.gen_cls_id_to_parent_cls_id(test_genome_class_id, rank))
