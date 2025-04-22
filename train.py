import pickle
from collections import defaultdict
import time

def get_pair_freqs(word_splits: dict, pre_tokens: dict):
    pair_freqs = defaultdict(int)
    pair_index = defaultdict(set)

    for word, freq in pre_tokens.items():
        split = word_splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_freqs[pair] += freq
            pair_index[pair].add(word)

    return pair_freqs, pair_index

def merge_pair(word_splits: dict, pair: tuple, pair_index: dict):
    words = pair_index[pair]
    for word in words:
        split = word_splits[word]
        if len(split) == 1:
            continue
        
        i = 0
        while i < len(split) - 1:
            if split[i] == pair[0] and split[i+1] == pair[1]:
                # print("Old", split)
                split = split[:i] + [pair[0] + pair[1]] + split[i+2:]
                # print("Now", split)
            else:
                i += 1
            
        word_splits[word] = split
    
    return word_splits

def update_pair_freqs(word_splits: dict, pre_tokens: dict, best_pair: tuple, pair_freqs: dict, pair_index: dict, new_vocab_token: bytes):
    words = pair_index[best_pair]
    pair_freqs[best_pair] = 0

    for word in words:
        split = word_splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            if pair[0] == new_vocab_token or pair[1] == new_vocab_token:
                pair_freqs[pair] += pre_tokens[word]
            pair_index[pair].add(word)

    return pair_freqs, pair_index

def train(pre_tokens: dict, max_vocab_size: int = 10_000):
    vocab = {bytes([i]): i for i in range(256)}
    vocab["<|endoftext|>".encode("utf-8")] = 256
    merges = {}

    num_merges = max_vocab_size - len(vocab)

    word_splits = {pre_token: [bytes([b]) for b in pre_token.encode("utf-8")] for pre_token in pre_tokens}
    pair_freqs, pair_index = get_pair_freqs(word_splits, pre_tokens)

    for i in range(num_merges):
        best_pair = max(pair_freqs, key=pair_freqs.get)

        vocab_id = len(vocab)
        merges[best_pair] = vocab_id
        # print(best_pair, vocab_id)

        # start = time.time()
        word_splits = merge_pair(word_splits, best_pair, pair_index)
        # end = time.time()
        # print(f"Time taken merging pair: {end - start} seconds")

        new_vocab_token = best_pair[0] + best_pair[1]

        vocab[vocab_id] = new_vocab_token
        # start = time.time()
        pair_freqs, pair_index = update_pair_freqs(word_splits, pre_tokens, best_pair, pair_freqs, pair_index, new_vocab_token)
        # end = time.time()
        # print(f"Time taken updating pair freqs: {end - start} seconds")

    return vocab, merges


if __name__ == "__main__":
    split = "valid"
    with open(f"tokenizer/pre_tokens_{split}.pkl", "rb") as f:
        pre_tokens = pickle.load(f)

    start = time.time()
    vocab, merges = train(pre_tokens, max_vocab_size=10_000)
    end = time.time()
    print(f"Time taken training: {end - start} seconds")

    cnt = 0
    for k, v in vocab.items():
        print(k, v)
        cnt += 1
        if cnt > 300:
            break

    cnt = 0
    for k, v in reversed(vocab.items()):
        print(k, v)
        cnt += 1
        if cnt > 100:
            break

    cnt = 0
    for k, v in merges.items():
        print(k[0], k[1], v)
        cnt += 1
        if cnt > 100:
            break

    with open(f"tokenizer/vocab_{split}.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(f"tokenizer/merges_{split}.pkl", "wb") as f:
        pickle.dump(merges, f)
