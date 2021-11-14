# Semantic Web Technology - Group 4

This repository contains the code that was used for the final project
for the course semantic web technology. As this project was to evaluate
the performance of the SP(B)ERT architecture with a BART-based encoder,
many parts of the code have been adapted from the [SPBERT repository](https://github.com/heraclex12/NLP2SPARQL).

Note that the early stopping class was adapted from [this implementation](https://github.com/Bjarten/early-stopping-pytorch).

## How to run
The program entrypoint is main.py and has 4 modes available:
```bash
python3 main.py --mode train | test | eval | exp
```
### Train mode
The train mode allows for the training of the network. The following
command line arguments should be provided:
```bash
--max_source_length (INT): The maximum token input sequence length

--max_target_length (INT): The maximum token output sequence length

--device (cuda | cpu): The device to be used for training

--beam (INT): How many candidate answers to generate via beam search, per input

--data_folder (PATH): The folder in which the data is contained

--data_filename (STRING): The name of the dataset for training

--output_folder (PATH):   A (non-existent) folder in which the output
                          models will be stored

--train_steps (INT):      How many epochs to train for

--dev_filename (STRING):  The name of the dev files for validation.
                          Should be located in the folder specified
                          at --data_folder

--do_early_stopping:  Setting this flag enables early stopping

--delta_es (FLOAT):   If early stopping is enabled, specifies the
                      minimum change to the validation loss which
                      counts as progress

--patience (INT):     If early stopping is enabled, specifies the number
                      of epochs to wait for change prior to stopping
```
Note that the data filenames should have either an .en or .sparql file
extension. The system is currently only set up for mapping natural language to
SPARQL queries. If you wish to train the system without validation calculations,
then call the program without the --do_early_stopping flag.


If one wishes to also calculate the BLEU score during runtime, provide
the following flag:
```bash
--do_eval: Enables BLEU calculation on the dev set
```

Keep in mind that this severely reduces the performance. In addition,
for BLEU scores to be calculated, dev files will have to be provided as above.

Additional arguments exist that allow the user to specify further training
parameters, but these are not required for execution. These are:
```bash
--weight_decay (FLOAT): Allows for weight decay to be applied
--gradient_accumulation_steps (INT):  How many epochs need to be collected in
                                      a gradient prior to loss calculation
--learning_rate (FLOAT): The learning rate for the AdamW optimizer.
--adam_epsilon (FLOAT): The epsilon for the AdamW optimizer.

```

### Test mode
The test mode takes in a test dataset and generates predictions. Generates two
output files: .answers and .topXanswers. The .answers file contains on each
line the answer to the input on the corresponding line of the test input file.
The .topXanswers provides the X most likely output sequences based on the input
file, where X is the beam size specified in the arguments. The arguments that
have to be provided are as follows:

```bash
--max_source_length (INT): The maximum token input sequence length

--max_target_length (INT): The maximum token output sequence length

--device (cuda | cpu): The device to be used for evaluation

--beam (INT): How many candidate answers to generate via beam search, per input

--data_folder (PATH): The folder in which the data is contained

--data_filename (STRING): The name of the dataset for testing

--model_params (PATH):  A path to the parameters of the trained network to
                        evaluate.
```
### Eval mode
The eval mode takes the answers generated by the network and a file with the
actual answers, and calculates the BLEU and EM scores. Note that the
outputs from the network in test mode first require minimal additional
pre-processing due to differences in spaces. See the space_mapper function
in util.py for that purpose.

```bash
--gold (PATH): The actual queries (golden standard)
--output (PATH): The predicted queries
```

### Exp mode
This mode was only used to experiment with seeing the answers to a single
query, and in general is not necessary except for development purposes.
