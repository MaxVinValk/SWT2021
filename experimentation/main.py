import torch
from tqdm import tqdm
from transformers import BartTokenizer, BertTokenizer,  BertModel
from custom_models import Bart2BertModel
from data_loader import get_loader


def franken_assemble():
    test = Bart2BertModel.from_pretrained("facebook/bart-base")

    spbert = BertModel.from_pretrained("razent/spbert-mlm-wso-base")
    spbert.config.is_decoder = True

    test.decoder = spbert.encoder

    return test


if __name__ == '__main__':
    NUM_TRAIN_EPOCHS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    spbert_tokenizer = BertTokenizer.from_pretrained("razent/spbert-mlm-wso-base")
    model = franken_assemble()

    data_loader = get_loader("data/train-data", "en", "sparql", bart_tokenizer, spbert_tokenizer)

    model.to(DEVICE)
    model.train()

    # TODO: Maybe the stuff prior to the loop
    for epoch in range(NUM_TRAIN_EPOCHS):
        bar = tqdm(data_loader, total=len(data_loader))

        for batch in bar:
            batch = tuple(t.to(DEVICE) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                               target_mask=target_mask)



