from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
import torch
from transformers.modeling_outputs import Seq2SeqModelOutput


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


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
        self.encoder = self.encoder.encoder
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_hf)

        # Create config, model and tokenizer for decoder
        self.decoder_config = AutoConfig.from_pretrained(self.decoder_hf)
        self.decoder_config.is_decoder = True
        self.decoder_config.add_cross_attention = True
        self.decoder = AutoModel.from_config(self.decoder_config)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(self.decoder_hf)

        # self.seq2seq.decoder = self.decoder.encoder

        # From their code
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.lsm = nn.LogSoftmax(dim=-1)

        # The weights of the last layer in the encoder to be the same as
        # the first layer in the decoder
        # self.tie_weights()

        self.beam_size = None
        self.max_length = None
        self.sos_id = None
        self.eos_id = None
        
    def freeze_params(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

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

    def forward_pass(self, input_sentence):
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

    def forward(
        self,
        source_ids=None,
        source_mask=None,
        target_ids=None,
        target_mask=None,
        args=None,
    ):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0]
        if target_ids is not None:
            out = self.decoder(
                input_ids=target_ids,
                attention_mask=target_mask,
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=source_mask,
            )
            hidden_states = torch.tanh(self.dense(out[0]))
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = LabelSmoothingLoss(self.config.vocab_size, smoothing=0.1)
            # loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                shift_labels.view(-1)[active_loss],
            )

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return outputs
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[i : i + 1, :]
                context_mask = source_mask[i : i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(self.beam_size, 1, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break

                    attn_mask = input_ids > 0
                    out = self.decoder(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        encoder_hidden_states=context,
                        encoder_attention_mask=context_mask,
                    )
                    hidden_states = torch.tanh(self.dense(out[0]))[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(
                        input_ids.data.index_select(0, beam.getCurrentOrigin())
                    )
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[: self.beam_size]
                pred = [
                    torch.cat(
                        [x.view(-1) for x in p] + [zero] * (self.max_length - len(p))
                    ).view(1, -1)
                    for p in pred
                ]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[: self.size - len(self.finished)]
        return self.finished[: self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
