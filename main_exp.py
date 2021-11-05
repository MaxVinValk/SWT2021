import torch
from data_loader import encode_single_example
from model import EncoderDecoder
from util import revert_query

def main_exp(args):
    # We initialize the model to generate 10 answers per input
    model = EncoderDecoder(device=args.device, beam_size=args.beam, max_length=args.max_target_length)
    model.load_state_dict(torch.load(args.model_params))

    model.eval()
    model.to(args.device)

    source_sentence = "How many movies did Stanley Kubrick direct?"
    target_sentence = "SELECT DISTINCT COUNT(?uri) WHERE {?uri <http://dbpedia.org/ontology/director> " \
                      "<http://dbpedia.org/resource/Stanley_Kubrick>  . } "

    source_ids, source_mask, _, _ = encode_single_example(
        source_sentence,
        target_sentence,
        model.encoder_tokenizer,
        model.decoder_tokenizer,
        args.max_source_length,
        args.max_target_length,
    )

    source_ids.to(args.device)
    source_mask.to(args.device)

    with torch.no_grad():
        res = model(source_ids, source_mask)

    given_results = []

    for i in range(len(res[0])):
        output = list(res[0][i].numpy())

        if 0 in output:
            output = output[: output.index(0)]

        sentence = model.decoder_tokenizer.decode(output, clean_up_tokenization_spaces=False)
        given_results.append(sentence)

    print(f"\n\nSource sentence: {source_sentence}")
    print(f"Actual target: {target_sentence}")

    print("----------\n")

    for i in range(len(given_results)):
        print(f"{i + 1}: {given_results[i]}")

        print("--------------")
        print(revert_query(given_results[i]))