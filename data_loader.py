import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler


class Example(object):
    """A single training/test example."""

    def __init__(
        self,
        idx,
        source,
        target,
    ):
        self.idx = idx
        self.source = source
        self.target = target


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        example_id,
        source_ids,
        target_ids,
        source_mask,
        target_mask,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def read_examples(source_file, target_file):
    """Read examples from filename."""
    examples = []
    with open(source_file, encoding="utf-8") as source_f:
        with open(target_file, encoding="utf-8") as target_f:
            for idx, (source, target) in enumerate(zip(source_f, target_f)):
                examples.append(
                    Example(
                        idx=idx,
                        source=source.strip(),
                        target=target.strip(),
                    )
                )
    return examples


def convert_examples_to_features(
    examples,
    tokenizer_source,
    tokenizer_target,
    max_source_length,
    max_target_length,
    stage=None,
):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer_source.tokenize(example.source)[
            : max_source_length - 2
        ]
        source_tokens = (
            [tokenizer_source.cls_token] + source_tokens + [tokenizer_source.sep_token]
        )
        source_ids = tokenizer_source.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer_source.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer_target.tokenize("None")
        else:
            target_tokens = tokenizer_target.tokenize(example.target)[
                : max_target_length - 2
            ]
        target_tokens = (
            [tokenizer_target.cls_token] + target_tokens + [tokenizer_target.sep_token]
        )
        target_ids = tokenizer_target.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer_target.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def get_loader(
    train_filename,
    source,
    target,
    tokenizer_source,
    tokenizer_target,
    max_source_length=64,
    max_target_length=64,
    batch_size=8,
    grad_accum_steps=1,
    stage="train",
):
    # TODO: Get all these arguments nice and neat in an arg parser again

    train_examples = read_examples(
        train_filename + "." + source, train_filename + "." + target
    )
    train_features = convert_examples_to_features(
        train_examples,
        tokenizer_source,
        tokenizer_target,
        max_source_length,
        max_target_length,
        stage=stage,
    )

    all_source_ids = torch.tensor(
        [f.source_ids for f in train_features], dtype=torch.long
    )
    all_source_mask = torch.tensor(
        [f.source_mask for f in train_features], dtype=torch.long
    )
    all_target_ids = torch.tensor(
        [f.target_ids for f in train_features], dtype=torch.long
    )
    all_target_mask = torch.tensor(
        [f.target_mask for f in train_features], dtype=torch.long
    )
    train_data = TensorDataset(
        all_source_ids, all_source_mask, all_target_ids, all_target_mask
    )

    if stage == "train":
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = SequentialSampler(train_data)

    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size // grad_accum_steps
    )

    return train_dataloader


def encode_single_example(source, target, tokenizer_source, tokenizer_target, max_source_length, max_target_length):
    ex = [Example(
        idx=0,
        source=source.strip(),
        target=target.strip(),
    )]

    f = convert_examples_to_features(ex, tokenizer_source, tokenizer_target,
        max_source_length, max_target_length, stage="train")

    encoded = [f[0].source_ids, f[0].source_mask, f[0].target_ids, f[0].target_mask]

    return tuple(torch.LongTensor([e]) for e in encoded)


def find_max_feature_lengths(train_filename, source, target, tokenizer_source, tokenizer_target):
    train_examples = read_examples(
        train_filename + "." + source, train_filename + "." + target
    )

    max_source = 0
    max_target = 0

    for example in train_examples:
        len_source = len(tokenizer_source.tokenize(example.source))
        len_target = len(tokenizer_target.tokenize(example.target))

        max_source = max(max_source, len_source)
        max_target = max(max_target, len_target)

    return max_source, max_target
