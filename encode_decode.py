import os
import pickle
import regex as re
from collections import defaultdict

class Tokenizer:
    def __init__(self, tokenizer_path: str, special_tokens: list[str] = ["<|endoftext|>"], split: str = "train"):
        vocab_path = os.path.join(tokenizer_path, f"vocab_{split}.pkl")
        merges_path = os.path.join(tokenizer_path, f"merges_{split}.pkl")
        vocab_special_tokens_path = os.path.join(tokenizer_path, f"vocab_special_tokens_{split}.pkl")
        
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.vocab = pickle.load(open(vocab_path, "rb"))
        self.vocab_special_tokens = pickle.load(open(vocab_special_tokens_path, "rb"))
        self.inv_vocab_special_tokens = {v: k for k, v in self.vocab_special_tokens.items()}
        self.merges = pickle.load(open(merges_path, "rb"))
        self.special_tokens = special_tokens
    
    def get_pair_freqs(self, pre_token_bytes: list[bytes]):
        pair_freqs = defaultdict(int)

        for i in range(len(pre_token_bytes) - 1):
            pair = (pre_token_bytes[i], pre_token_bytes[i+1])
            pair_freqs[pair] += 1

        return pair_freqs

    def merge_pair(self, bytes_arr: list[bytes], pair: tuple, tokens: list[int], new_token: int):          
        i = 0
        while i < len(bytes_arr) - 1:
            if bytes_arr[i] == pair[0] and bytes_arr[i+1] == pair[1]:
                bytes_arr = bytes_arr[:i] + [pair[0] + pair[1]] + bytes_arr[i+2:]
                tokens = tokens[:i] + [new_token] + tokens[i+2:]
            else:
                i += 1
            
        return bytes_arr, tokens

    def encode_pre_token(self, pre_token: bytes) -> list[int]:
        bytes_arr = [bytes([b]) for b in pre_token]
        tokens = list(pre_token)

        while len(bytes_arr) > 2:
            pair_freqs = self.get_pair_freqs(bytes_arr)
            best_pair = min(pair_freqs, key=lambda x: self.merges.get(x, float("inf")))

            if best_pair not in self.merges:
                break

            new_token = self.merges[best_pair]

            bytes_arr, tokens = self.merge_pair(bytes_arr, best_pair, tokens, new_token)

        return tokens

    def encode(self, text: str) -> list[int]:
        special_token_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        text_chunks = re.split(special_token_pattern, text)
        tokens = []
        for text_chunk in text_chunks:
            chunk = text_chunk.encode("utf-8")
            if chunk in self.vocab_special_tokens:
                tokens.append(self.vocab_special_tokens[chunk])
            else:
                for match in re.finditer(self.pattern, text_chunk):
                    pre_token = match.group(0)
                    curr_tokens = self.encode_pre_token(pre_token.encode("utf-8"))
                    tokens.extend(curr_tokens)
        return tokens

    def decode(self, tokens: list[int], print_tokenwise: bool = False) -> str:
        bytes_str = b""
        for token in tokens:
            if token in self.vocab:
                bytes_str += self.vocab[token]
                if print_tokenwise:
                    print(self.vocab[token])
            elif token in self.inv_vocab_special_tokens:
                bytes_str += self.inv_vocab_special_tokens[token]
                if print_tokenwise:
                    print(self.inv_vocab_special_tokens[token])
            else:
                raise ValueError(f"Token {token} not found in vocab or special tokens")
        
        return bytes_str.decode("utf-8", errors="replace")
        

if __name__ == "__main__":
    tokenizer = Tokenizer("tokenizer/", split="valid")
    s = "Hello, world! <|endoftext|>\nHow are you? Excited?"
    toks = tokenizer.encode(s)

    print("Original String")
    print("---------------")
    print(s)

    print("\nEncoded Tokens")
    print("---------------")
    print(toks)

    print("\nDecoded String")
    print("---------------")
    print(tokenizer.decode(toks, print_tokenwise=True))