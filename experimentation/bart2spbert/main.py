from transformers import BartTokenizer,  BertModel
from custom_models import Bart2BertModel

if __name__ == '__main__':
    bt = BartTokenizer.from_pretrained("facebook/bart-base")
    test = Bart2BertModel.from_pretrained("facebook/bart-base")

    spbert = BertModel.from_pretrained("razent/spbert-mlm-wso-base")
    spbert.config.is_decoder = True

    test.decoder = spbert.encoder

    test_sentence = "This is a test encoding in BARTs tokenizer"
    inputs = bt(test_sentence, return_tensors="pt")

    outputs = test(**inputs)

    print(outputs)
