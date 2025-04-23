# Byte-Pair Encoding Tokenization

An implementation of Byte-Pair Encoding (BPE) tokenization on a sample of the TinyStories dataset.

The main compoonents are split into inidividual files for better understanding.

1. `pre_tokenize.py` - Pre-tokenizes the text into tokens. Ensures that the text is split at the correct boundaries (special tokens). Uses the GPT2 regex pattern for pre-tokenization. Stores the frequency of the pre-tokenized tokens in a pickle file.
2. `train.py` - Trains the BPE model on the pre-tokenized text. Stores the vocabulary and merges in pickle files.
3. `encode_decode.py` - Encodes and decodes text using the BPE model using the vocabulary and merges.

## Usage

```bash
python pre_tokenize.py
python train.py
python encode_decode.py
```

## Obtaining the data

Instructions from the [Stanford CS336 assignment](https://github.com/stanford-cs336/assignment1-basics):

```
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```