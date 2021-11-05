import torch
from tqdm import tqdm
from model import EncoderDecoder
from data_loader import get_loader, encode_single_example
from transformers import AdamW, get_linear_schedule_with_warmup
from util import get_argparser, set_seed, save_model, revert_query
from main_es_eval import main_es_eval


def main_train(args):
    set_seed(args.seed)

    model = EncoderDecoder(device=args.device, beam_size=args.beam, max_length=args.max_target_length)

    model.freeze_params()
    model.to(args.device)
    model.train()

    data_loader = get_loader(
        f"{args.data_folder}/{args.data_filename}",
        "en",
        "sparql",
        model.encoder_tokenizer,
        model.decoder_tokenizer,
    )

    # The parameters we do not wish to optimize
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    t_total = len(data_loader) // args.gradient_accumulation_steps * args.train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(t_total * 0.1),
                                                num_training_steps=t_total)

    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6

    for epoch in range(args.train_steps):
        bar = tqdm(data_loader, total=len(data_loader))

        for batch in bar:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _ = model(
                source_ids=source_ids,
                source_mask=source_mask,
                target_ids=target_ids,
                target_mask=target_mask,
            )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            tr_loss += loss.item()
            train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
            bar.set_description("epoch {} loss {}".format(epoch, train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            # Update parameters
            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

    save_model(model, f"{args.output_folder}/final")


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


def main_pred(args):
    model = EncoderDecoder(device=args.device, beam_size=args.beam, max_length=args.max_target_length)
    model.load_state_dict(torch.load(args.model_params))

    model.eval()
    model.to(args.device)





if __name__ == "__main__":
    args = get_argparser().parse_args()

    if args.mode == "train":
        main_train(args)

        # Uncomment for ES and BLEU eval suppoting main
        # main_es_eval(args)
    else:
        main_exp(args)
