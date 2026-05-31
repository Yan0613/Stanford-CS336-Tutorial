import regex as re
import os
from typing import BinaryIO
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _to_byte_tuple(s: str) -> tuple[bytes, ...]:
    b = s.encode("utf-8")
    return tuple(b[i:i+1] for i in range(len(b)))


def _build_special_pattern(special_tokens: list[str]) -> str:
    return "|".join(f"(?:{re.escape(token)})" for token in special_tokens)

def _count_pairs(pre_token_freq: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_freq = Counter()
    for pre_token, freq in pre_token_freq.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i+1])
            pair_freq[pair] += freq
    return pair_freq

def _init_indices(pre_token_freq: dict[tuple[bytes, ...], int]) -> tuple[
    list[list[bytes]],
    list[int],
    Counter,
    dict[tuple[bytes, bytes], set[int]],
]:
    pre_tokens = []
    pre_token_count = []
    pair_freq = Counter()
    pair_to_pretokens = {}                   

    for pre_token, freq in pre_token_freq.items():
        pre_tokens.append(list(pre_token))              
        pre_token_count.append(freq)
        idx = len(pre_tokens) - 1             

        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i+1])
            pair_freq[pair] += freq
            pair_to_pretokens.setdefault(pair, set()).add(idx)

    return pre_tokens, pre_token_count, pair_freq, pair_to_pretokens


"""
non - streaming version of _apply_merge
"""
def _apply_merge(pre_token_freq: dict[tuple[bytes, ...], int], merge: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    A, B = merge
    new_pre_token_freq = Counter()
    
    for pre_token, freq in pre_token_freq.items():
        # ===== 扫一遍 pre_token,把相邻的 (A, B) 合并 =====
        new_pre_token = []
        i = 0
        while i < len(pre_token):
            if pre_token[i:i+2] == (A, B):
                new_pre_token.append(A + B)
                i += 2
            else:
                new_pre_token.append(pre_token[i])
                i += 1
        new_pre_token_freq[tuple(new_pre_token)] += freq
    return new_pre_token_freq




def _apply_merge_incremental(
    pair: tuple[bytes, bytes],                          
    pre_tokens: list[list[bytes]],                      
    pre_token_count: list[int],                        
    pair_freq: Counter,                                 
    pair_to_pretokens: dict[tuple[bytes, bytes], set[int]]) -> None:  
    """
    "对所有包含 pair=(A,B) 的 pre-token 做一次合并,
    并就地更新 pair_freq 和 pair_to_pretokens。
    """
    A, B = pair
    new_token = A + B

    affected_idxs = pair_to_pretokens[pair].copy()
    for idx in affected_idxs:
        seq = pre_tokens[idx]
        freq = pre_token_count[idx]

        old_pairs = Counter()
        for i in range(len(seq) - 1):
            old_pairs[seq[i], seq[i+1]] += 1

        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == A and seq[i+1] == B:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        pre_tokens[idx] = new_seq  # 就地更新 pre_tokens

        new_pairs = Counter()
        for i in range(len(new_seq) - 1):
            new_pairs[new_seq[i], new_seq[i+1]] += 1
        
        all_pairs_involved = set(old_pairs) | set(new_pairs)
        for p in all_pairs_involved:
            delta = new_pairs.get(p, 0) - old_pairs.get(p, 0)
            if delta != 0:
                pair_freq[p] += delta * freq
                if pair_freq[p] <= 0:
                    del pair_freq[p]
            if p in new_pairs:
                pair_to_pretokens.setdefault(p, set()).add(idx)
            else:
                if p in pair_to_pretokens:
                    pair_to_pretokens[p].discard(idx)
                    if not pair_to_pretokens[p]:
                        del pair_to_pretokens[p]



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str])->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # num_processes = os.cpu_count()
    special_pattern = _build_special_pattern(special_tokens)
    
    pre_token_freq: Counter[tuple[bytes, ...]] = Counter()
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1, special_tokens[0].encode())
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            sub_chunks = re.split(special_pattern, chunk) if special_pattern else [chunk]
            for sub_chunk in sub_chunks:
                if not sub_chunk:
                    continue
                for m in re.finditer(PAT, sub_chunk):
                    pre_token_freq[_to_byte_tuple(m.group())] += 1

    
    # ===== main BPE run loop =====
    # initialize the indices
    pre_tokens, pre_token_count, pair_freq, pair_to_pretokens = _init_indices(pre_token_freq)

    vocab = {i: bytes([i]) for i in range(256)}
    for k, st in enumerate(special_tokens):
        vocab[256 + k] = st.encode("utf-8")
    
    merges = []
    while len(vocab) < vocab_size:
        # not streaming version
        #pair_freq = _count_pairs(pre_token_freq)
        if not pair_freq:
            break
        best_pair = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))[0]
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)
        _apply_merge_incremental(best_pair, pre_tokens, pre_token_count, pair_freq, pair_to_pretokens)
    
    return vocab, merges
    

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]| None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        #indices
        self.bytes_to_id : dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge_rank: dict[tuple[bytes, bytes], int] = {m: i for i, m in enumerate(merges)}
        self.special_tokens_sorted = sorted(self.special_tokens, key = len, reverse = True)
    
    def _encode_pretoken(self, pre_token: bytes) -> list[int]:
        parts = [bytes([b]) for b in pre_token]

        while True:
            best_rank = float('inf')
            best_pair = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                rank = self.merge_rank.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            # 一次扫描,合并所有相邻 (A, B)
            A, B = best_pair
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == A and parts[i+1] == B:
                    new_parts.append(A + B)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        return [self.bytes_to_id[p] for p in parts]

    def encode(self, text: str) -> list[int]:
        token_ids = []
        
        # ----- 第 1 步:special_tokens 切大段 -----
        if self.special_tokens_sorted:
            # 注意 1:re.escape 防止 <| 等被当成正则元字符
            # 注意 2:用 () 包起来,re.split 才会保留 special token(否则会被丢掉)
            # 注意 3:按长度倒序(已经在 __init__ 排过序),长的优先匹配
            special_pat = "(" + "|".join(re.escape(t) for t in self.special_tokens_sorted) + ")"
            segments = re.split(special_pat, text)
        else:
            segments = [text]
        
        # ----- 第 2 步:遍历 segments -----
        for seg in segments:
            if not seg:
                continue
            if seg in self.special_tokens:
                # special token:直接查 vocab
                token_ids.append(self.bytes_to_id[seg.encode("utf-8")])
            else:
                # 普通段:先 pre-tokenize,再每个 pre-token 跑 BPE
                for m in re.finditer(PAT, seg):
                    pre_token_bytes = m.group().encode("utf-8")
                    token_ids.extend(self._encode_pretoken(pre_token_bytes))
        
        return token_ids

    def decode(self, tokens: list[int]) -> str:
        return b''.join(self.vocab[t] for t in tokens).decode("utf-8", errors="replace")

    
    def encode_iterable(self, iterable):
        """
        流式编码:接收一个可迭代对象(通常是打开的文件),
        每行调用一次 encode,逐个 yield 出 token id。
        """
        for line in iterable:
            for tid in self.encode(line):
                yield tid