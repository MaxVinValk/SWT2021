from nltk.translate.bleu_score import corpus_bleu
from nltk.metrics import accuracy
import argparse


def preprocess(f):
    """Makes a list of lines, where each line is split up by word"""
    return [line.split() for line in f]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", dest="gold", required=True)
    parser.add_argument("--output", dest="output", required=True)
    args = parser.parse_args()

    # Process files
    gold_em = preprocess(open(args.gold, "r"))
    # BLEU expects a list of golden standards
    gold_bleu = [[i] for i in gold_em]
    output = preprocess(open(args.output, "r"))

    # Calculate & print results
    bleu = corpus_bleu(gold_bleu, output)
    em = accuracy(gold_em, output)

    print(f"BLEU score: {bleu}")
    print(f"EM score: {em}")


if __name__ == "__main__":
    main()
