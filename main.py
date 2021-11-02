import torch
from tqdm import tqdm
from model import EncoderDecoder
from data_loader import get_loader
from transformers import AdamW, get_linear_schedule_with_warmup

NUM_TRAIN_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = EncoderDecoder()
    model.freeze_params()
    # frozen_layers = 0
    # for name, p in model.named_parameters():
    #     if name.startswith("encoder"):
    #         p.requires_grad = False
    #         frozen_layers += 1
    #     else:
    #         break
    # print(f"Froze {frozen_layers} params of the encoder.")

    data_loader = get_loader(
        "data/train-data",
        "en",
        "sparql",
        model.encoder_tokenizer,
        model.decoder_tokenizer
    )

    model.to(DEVICE)
    model.train()

    # The parameters we do not wish to optimize
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = len(data_loader) // 1 * NUM_TRAIN_EPOCHS
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(t_total * 0.1),
                                                num_training_steps=t_total)

    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6

    # TODO: Maybe the stuff prior to the loop
    for epoch in range(NUM_TRAIN_EPOCHS):
        bar = tqdm(data_loader, total=len(data_loader))

        for batch in bar:
            batch = tuple(t.to(DEVICE) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _ = model(
                source_ids=source_ids,
                source_mask=source_mask,
                target_ids=target_ids,
                target_mask=target_mask,
            )

            tr_loss += loss.item()
            train_loss = round(tr_loss / (nb_tr_steps + 1), 4)
            bar.set_description("epoch {} loss {}".format(epoch, train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

