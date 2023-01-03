import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


plt.style.use('seaborn')


def main():
    parser = argparse.ArgumentParser(description="Creates a plot that visualizes l2-distances")
    parser.add_argument("--out-folder", dest="out_folder", type=str, help="Path where to save image", required=True)
    parser.add_argument("--transparent", dest="transparent", action="store_true", help="Whether image should have transparent background or not", default=False)
    args = parser.parse_args()
    transparent = args.transparent
    out_folder = args.out_folder

    # Results should be entered here, second element of tuple will be printed in second line of each xlabel
    error_collection = [
        ("EmbedLstmAtt.", "", 0.0019008730231133265),
        ("Tf Vocab", r'$k=12$', 0.002044093517511694),
        ("Tf Vocab", r'$k=13$', 0.0013693966372092486),
        ("Tf Vocab ML", r'$k=12$', 0.0026541913536266863),
        ("Tf Lsh", r'$b=2^{22}$', 0.0030668235752430674),
        ("Tf Lsh", r'$b=2^{23}$', 0.0025782383973041233),
        ("Tf Hash-embed", "", 0.001910600756306769),
        ("Kraken2", "", 0.0050686464)
    ]

    names, modifications, vals = list(list(x) for x in zip(*error_collection))
    labels = [name + "\n" + modification for name, modification in zip(names, modifications)]
    xticks = np.arange(0, len(labels))

    fig, axes = plt.subplots()
    axes.bar(labels, vals)

    axes.set_xticks(xticks)
    axes.set_xticklabels(labels)
    axes.set_xlabel("Model", labelpad=10)
    axes.set_ylabel("Error (L2 distance)", labelpad=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "abundance_error.pdf"), transparent=transparent)


if __name__ == "__main__":
    main()