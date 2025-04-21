import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import multiprocessing
import time
import pickle

NUM_PROCESSES = 128

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_chunks(file_path: str, split_special_token: bytes) -> list[str]:
    with open(file_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, NUM_PROCESSES, split_special_token)
        print(chunk_boundaries)

        chunks = []
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_post_special_tokens_split = [c for c in chunk.split(split_special_token.decode("utf-8")) if c]
            chunks.extend(chunk_post_special_tokens_split)

        print(len(chunks))
        return chunks

def pretokenize_chunk(pattern: re.Pattern, chunk: str) -> list[str]:
    tokens_iter = re.finditer(pattern, chunk)
    dc = defaultdict(int)
    for token in tokens_iter:
        dc[token.group(0)] += 1

    return dc

def pretokenize(pre_tokenization_regex: str, file_path: str, split_special_token: bytes) -> dict:
    pattern = re.compile(pre_tokenization_regex)
    chunks = get_chunks(file_path, split_special_token)
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = pool.starmap(pretokenize_chunk, [(pattern, chunk) for chunk in chunks])

    final_merged_results = defaultdict(int)
    for dc in results:
        for token, count in dc.items():
            final_merged_results[token] += count

    return final_merged_results


if __name__ == "__main__":
    start = time.time()
    pre_tokens = pretokenize(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", "data/TinyStoriesV2-GPT4-train.txt", "<|endoftext|>".encode("utf-8"))
    print(len(pre_tokens))
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    
    cnt = 0
    for pre_token, count in sorted(pre_tokens.items(), key=lambda x: x[1], reverse=True):
        print(f"{pre_token}: {count}")
        cnt += 1
        if cnt > 100:
            break

    print(pre_tokens[" the"])

    # Store the pre_tokens in a pickle file
    with open("pre_tokens.pkl", "wb") as f:
        pickle.dump(pre_tokens, f)