from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqModelOutput


class EncoderDecoder(nn.Module):
    def __init__(
        self, encoder_hf="facebook/bart-base", decoder_hf="razent/spbert-mlm-wso-base"
    ) -> None:
        super(EncoderDecoder, self).__init__()

        self.encoder_hf = encoder_hf
        self.decoder_hf = decoder_hf

        # Create config, model and tokenizer for encoder
        self.config = AutoConfig.from_pretrained(self.encoder_hf)
        # This is the encoder but technically BART has encoder and decoder, so we set
        #   a name "seq2seq", because we change the decoder to the desired one: SPBERT.
        self.encoder = AutoModel.from_config(self.config)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_hf)

        # Create config, model and tokenizer for decoder
        self.decoder_config = AutoConfig.from_pretrained(self.decoder_hf)
        self.decoder_config.is_decoder = True
        self.decoder_config.add_cross_attention = True
        self.decoder = AutoModel.from_config(self.decoder_config)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(self.decoder_hf)

        # self.seq2seq.decoder = self.decoder.encoder

        self.training = False
        self.decoder.training = False
        self.encoder.training = False

    def _forward_encoder(self, input_sentence):
        tokenized_input = self.encoder_tokenizer(input_sentence, return_tensors="pt")
        input_ids = tokenized_input["input_ids"]
        attention_mask = tokenized_input["attention_mask"]

        encoder_output = self.encoder.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return encoder_output, attention_mask

    def _forward_decoder(
        self, encoder_output, decoder_attention_mask=None, encoder_attention_mask=None
    ):
        # var1 = encoder_output.last_hidden_state
        # input_shape = var1.size()
        return self.decoder.encoder(
            encoder_output.last_hidden_state,
            encoder_hidden_states=encoder_output[0],
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    def forward(self, input_sentence):
        encoder_output, encoder_attention_mask = self._forward_encoder(input_sentence)
        decoder_output = self._forward_decoder(
            encoder_output=encoder_output, encoder_attention_mask=encoder_attention_mask
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_output.last_hidden_state,
            past_key_values=decoder_output.past_key_values,
            decoder_hidden_states=decoder_output.hidden_states,
            decoder_attentions=decoder_output.attentions,
            cross_attentions=decoder_output.cross_attentions,
            encoder_last_hidden_state=encoder_output.last_hidden_state,
            encoder_hidden_states=encoder_output.hidden_states,
            encoder_attentions=encoder_output.attentions,
        )


encoder_decoder = EncoderDecoder()
res = encoder_decoder("This is a test encoding in BARTs tokenizer")

print(res)
