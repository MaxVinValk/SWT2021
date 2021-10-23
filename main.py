import torch
from tqdm import tqdm
from model import EncoderDecoder
from data_loader import get_loader

NUM_TRAIN_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = EncoderDecoder()

    data_loader = get_loader(
        "data/train_preprocessed",
        "en",
        "sparql",
        model.encoder_tokenizer,
        model.decoder_tokenizer
    )

    model.to(DEVICE)
    model.train()

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
            print(loss)
