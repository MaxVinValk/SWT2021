from provided import seperate_en_sparql, generator_utils, preprocessing
import argparse, os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data_dir", required=True)
    parser.add_argument("--subset", dest="subset", required=True)
    parser.add_argument("--output", dest="output_dir", required=True)
    args = parser.parse_args()

    if not os.path.isfile(
        f"{args.output_dir}/{args.subset}.sparql"
    ) or not os.path.isfile(f"{args.output_dir}/{args.subset}.en"):
        print("Seperating data...")
        seperate_en_sparql.seperate(args.data_dir, args.subset, args.output_dir)

    if not os.path.isfile(
        f"{args.output_dir}/{args.subset}_preprocessed.sparql"
    ) or not os.path.isfile(f"{args.output_dir}/{args.subset}_preprocessed.en"):
        print("Preprocessing data...")
        preprocessing.preprocesser(
            f"{args.output_dir}/{args.subset}.en",
            f"{args.output_dir}/{args.subset}.sparql",
            f"{args.output_dir}/{args.subset}_preprocessed",
        )


if __name__ == "__main__":
    main()
