import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import multiprocessing
import time
import pickle

NUM_PROCESSES = 256

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently based on byte offsets.
    Ensures chunks do not split the `split_special_token`.
    May return fewer chunks than desired if boundaries overlap.
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

def process_chunk(
    file_path: str, 
    start: int, 
    end: int, 
    regex_pattern_str: str, 
    split_special_token_bytes: bytes
) -> defaultdict[str, int]:
    """
    Reads a specific byte range, decodes it, splits by the special token,
    and counts token occurrences in each sub-chunk using the regex pattern.
    """
    pattern = re.compile(regex_pattern_str)
    counts = defaultdict(int)
    try:
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
            
            split_token_str = split_special_token_bytes.decode("utf-8", errors="ignore")

            sub_chunks = chunk_str.split(split_token_str)

            # Process each sub-chunk (text between special tokens) individually
            for sub_chunk in sub_chunks:
                if not sub_chunk: # Skip empty strings resulting from split
                    continue
                # Find and count all token occurrences in the sub-chunk
                for match in re.finditer(pattern, sub_chunk):
                    token = match.group(0)
                    counts[token] += 1

    except FileNotFoundError:
        print(f"Error: File not found at {file_path} in worker process.")
    except Exception as e:
        print(f"Error processing chunk {start}-{end} in {file_path}: {e}")
    return counts

def pretokenize(pre_tokenization_regex: str, file_path: str, split_special_token: bytes) -> dict:
    """
    Finds chunk boundaries, processes chunks in parallel (splitting internally by special token), 
    and aggregates token counts.
    """
    # 1. Find chunk boundaries based on the special token
    try:
        with open(file_path, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(f, NUM_PROCESSES, split_special_token)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}

    print(f"Found {len(chunk_boundaries) - 1} chunks based on boundaries: {chunk_boundaries}")

    tasks = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        if start < end: # Ensure chunk has size > 0
            tasks.append((file_path, start, end, pre_tokenization_regex, split_special_token)) # Add split_special_token here
        else:
            print(f"Skipping empty chunk at boundary {start}")

    if not tasks:
        print("No tasks generated for processing.")
        return {}

    final_merged_results = defaultdict(int)
    try:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            results = pool.starmap(process_chunk, tasks)

        for dc in results:
            for token, count in dc.items():
                final_merged_results[token] += count

    except Exception as e:
        print(f"Error during multiprocessing or result merging: {e}")
        return dict(final_merged_results)

    return dict(final_merged_results) # Convert back to dict if needed, though defaultdict is fine


if __name__ == "__main__":
    split = "valid"
    start = time.time()
    pre_tokens = pretokenize(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", 
        f"data/TinyStoriesV2-GPT4-{split}.txt",
        b"<|endoftext|>" # Pass the byte string directly
    )
    print(len(pre_tokens))
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    
    cnt = 0
    for pre_token, count in sorted(pre_tokens.items(), key=lambda x: x[1], reverse=True):
        print(f"{pre_token}: {count}")
        cnt += 1
        if cnt > 100:
            break

    lookup_token = " the"
    if lookup_token in pre_tokens:
      print(f"\nCount for '{lookup_token}': {pre_tokens[lookup_token]}")
    else:
      print(f"\nToken '{lookup_token}' not found.")

    # Store the pre_tokens in a pickle file
    with open(f"tokenizer/pre_tokens_{split}.pkl", "wb") as f:
        pickle.dump(pre_tokens, f)