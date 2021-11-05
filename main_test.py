from model import EncoderDecoder
from data_loader import get_loader
import torch
from tqdm import tqdm
from util import revert_query


def main_test(args):
    model = EncoderDecoder(device=args.device, beam_size=args.beam, max_length=args.max_target_length)
    model.load_state_dict(torch.load(args.model_params))

    model.eval()
    model.to(args.device)

    # Load in dataset
    data_loader = get_loader(
        f"{args.data_folder}/{args.data_filename}",
        "en",
        "sparql",
        model.encoder_tokenizer,
        model.decoder_tokenizer,
        batch_size=1,
        stage="test"
    )

    bar = tqdm(data_loader, total=len(data_loader))

    total_answers = []

    for query in bar:
        query = tuple(t.to(args.device) for t in query)

        source_ids, source_mask, _, _ = query

        with torch.no_grad():
            res = model(source_ids, source_mask)

        answers = []
        for i in range(args.beam):
            output = list(res[0][i].numpy())

            if 0 in output:
                output = output[: output.index(0)]

            sentence = model.decoder_tokenizer.decode(output, clean_up_tokenizeation_spaces=False)
            answers.append(sentence)

        total_answers.append(revert_query(answers))

    # write to file
    with open(f"{args.data_folder}/{args.data_filename}.answers", "w") as f:
        for i in range(len(total_answers)):
            f.write(f"{total_answers[i][0]}\n")

    # write to file
    with open(f"{args.data_folder}/{args.data_filename}.top{args.beam}answers", "w") as f:
        for i in range(len(total_answers)):
            for j in range(len(total_answers[i])):
                f.write(f"{total_answers[i][j]}\n")

