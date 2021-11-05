import os
import random
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from early_stopping import EarlyStopping
from model import EncoderDecoder
from data_loader import convert_examples_to_features, get_loader, read_examples
from transformers import AdamW, get_linear_schedule_with_warmup
from util import save_model
from nltk.translate.bleu_score import corpus_bleu
import re


def main_es_eval(args):

    model = EncoderDecoder(device=args.device, beam_size=args.beam, max_length=args.max_target_length)

    model.freeze_params()

    data_loader = get_loader(
        f"{args.data_folder}/{args.data_filename}",
        "en",
        "sparql",
        model.encoder_tokenizer,
        model.decoder_tokenizer,
    )

    eval_data_loader = get_loader(
        f"{args.data_folder}/{args.dev_filename}",
        "en",
        "sparql",
        model.encoder_tokenizer,
        model.decoder_tokenizer,
    )

    model.to(args.device)
    model.train()

    # The parameters we do not wish to optimize
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, delta=float(
        args.delta_es), path=os.path.join(args.output_folder, "checkpoint-best-es"))

    t_total = len(
        data_loader) // args.gradient_accumulation_steps * args.train_steps
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
    )

    dev_dataset = {}
    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = (
        0,
        0,
        0,
        0,
        0,
        1e6,
    )

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

        if args.do_early_stopping:
            model.eval()
            eval_loss = 0
            eval_bar = tqdm(eval_data_loader, total=len(eval_data_loader))
            for batch in eval_bar:
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

                eval_loss += loss.item()

            early_stopping(eval_loss / len(eval_data_loader), model, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            model.train()

        # Eval on every epoch using BLEU
        if args.do_eval:
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            eval_flag = False
            if 'dev_bleu' in dev_dataset:
                eval_examples, eval_data = dev_dataset['dev_bleu']
            else:
                eval_examples = read_examples(
                    f"{args.data_folder}/{args.dev_filename}.{args.source}",
                    f"{args.data_folder}/{args.dev_filename}.{args.target}",
                )

                eval_examples = random.sample(
                    eval_examples, min(1000, len(eval_examples)))
                eval_features = convert_examples_to_features(
                    eval_examples, model.encoder_tokenizer, model.decoder_tokenizer, 64, 64, stage="test"
                )
                all_source_ids = torch.tensor(
                    [f.source_ids for f in eval_features], dtype=torch.long)
                all_source_mask = torch.tensor(
                    [f.source_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_source_ids, all_source_mask)
                dev_dataset['dev_bleu'] = eval_examples, eval_data

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            p = []
            for batch in eval_dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids,
                                  source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[: t.index(0)]
                        text = model.decoder_tokenizer.decode(
                            t, clean_up_tokenization_spaces=False
                        )
                        p.append(text)

            model.train()
            predictions = []
            pred_str = []
            label_str = []

            with open(os.path.join(args.output_folder, "dev.output"), "w") as f, open(
                os.path.join(args.output_folder, "dev.gold"), "w"
            ) as f1:
                for ref, gold in zip(p, eval_examples):
                    ref = ref.strip().replace("< ", "<").replace(" >", ">")
                    ref = re.sub(
                        r' ?([!"#$%&\'(’)*+,-./:;=?@\\^_`{|}~]) ?', r"\1", ref
                    )
                    ref = ref.replace("attr_close>", "attr_close >").replace(
                        "_attr_open", "_ attr_open"
                    )
                    ref = ref.replace(" [ ", " [").replace(" ] ", "] ")
                    ref = ref.replace("_obd_", " _obd_ ").replace(
                        "_oba_", " _oba_ "
                    )

                    pred_str.append(ref.split())
                    label_str.append([gold.target.strip().split()])
                    predictions.append(str(gold.idx) + "\t" + ref)
                    f.write(str(gold.idx) + "\t" + ref + "\n")
                    f1.write(str(gold.idx) + "\t" + gold.target + "\n")

            bl_score = corpus_bleu(label_str, pred_str) * 100

            print("  %s = %s " % ("BLEU", str(round(bl_score, 4))))
            print("  " + "*" * 20)
            if bl_score > best_bleu:
                print("  Best bleu: {}".format(bl_score))
                print("  " + "*" * 20)
                best_bleu = bl_score

                # Save best checkpoint for best bleu
                output_dir = os.path.join(
                    args.output_folder, "checkpoint-best-bleu")
                save_model(model, output_dir)
