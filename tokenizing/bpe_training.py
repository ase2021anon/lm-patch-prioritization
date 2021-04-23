import argparse
import tqdm
from pickle import load, dump
from collections import defaultdict, Counter

from bpe_utils import *

def count_tokens(data_file):
    with open(data_file) as f:
        all_cmds = f.readlines()
    
    all_cmds = [cmd.strip().lower() for cmd in all_cmds]
    all_tokens = []
    for cmd in all_cmds:
        all_tokens += cmd.split()
    
    d = Counter(all_tokens)
    tokens = list(d.keys())
    count_vs = [d[k] for k in tokens]
    all_tok_chars = [list(t) + [EOT] for t in tokens]
    return all_tok_chars, count_vs

def get_all_chars(all_tok_chars):
    all_char_freq = Counter()
    for char_list in tqdm.tqdm(all_tok_chars):
        indiv_char_counter = Counter(char_list)
        all_char_freq.update(indiv_char_counter)
    return all_char_freq

def count_pairs(args):
    print('Initializing...')
    all_tok_chars, count_vs = count_tokens(args.raw_file)
    all_char_freq = get_all_chars(all_tok_chars)
    char2idx = {c:i for i, c in enumerate(all_char_freq.keys())}
    all_bpe_ops = {}

    new_char_idx = len(all_char_freq)-1
    subst_cmds = [wrap_sep(TMP_SEP_TOK.join(char_cmd)) for char_cmd in all_tok_chars]
    print('Initialization complete. Merging...')
    while new_char_idx < args.vocab_size:
        # count all pairs in corpus (count unique tokens at once for speedup)
        all_duet_freq = defaultdict(int)
        for char_list, tok_count in zip(all_tok_chars, count_vs):
            for pair in zip(char_list, char_list[1:]):
                all_duet_freq[pair] += tok_count

        # find max pairs
        max_pairs = sorted(all_duet_freq.keys(), key=all_duet_freq.__getitem__, reverse=True)[:args.speedup]
        used_chars = set()

        # merge
        for max_pair in max_pairs:
            # make sure the pair we are merging is ok to merge
            if len((set(max_pair) | used_chars) - used_chars) < len(set(max_pair)):
                continue # if pair uses already used chars, don't merge
            else:
                used_chars |= set(max_pair)
            new_char_idx += 1

            # actual merge
            rep_str, new_str = wrap_sep(TMP_SEP_TOK.join(max_pair)), wrap_sep(''.join(max_pair))
            subst_cmds = [cmd.replace(rep_str, new_str) for cmd in subst_cmds]
            subst_cmds = [cmd.replace(rep_str, new_str) for cmd in subst_cmds] # fix overlap issue
            char2idx[unwrap_sep(new_str)] = new_char_idx
            all_bpe_ops[new_char_idx-len(all_char_freq)] = max_pair
            
            # track progress
            print(f'[{new_char_idx}/{args.vocab_size}] {max_pair} (count={all_duet_freq[max_pair]})')
        
        all_tok_chars = [unwrap_sep(cmd).split(TMP_SEP_TOK) for cmd in subst_cmds]
    
    print('')
    return char2idx, all_bpe_ops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learns BPE tokens from data.')
    parser.add_argument('--raw_file', help='File on which BPE shall be trained.')
    parser.add_argument('--vocab_size', type=int, default=5000,
                        help='How many byte-code pairs to extract.')
    parser.add_argument('--target_file_prefix', type=str, default='./train',
                        help='Prefix of saved vocabulary and merge operator files.')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Increases speed of pair extraction by allowing approximation.')
                        
    args = parser.parse_args()
    vocab, bpe_ops = count_pairs(args)

    with open(args.target_file_prefix + '_vocab_BPE.pkl', 'wb') as f:
        dump(char2idx, f)

    with open(args.target_file_prefix + '_varpairs_BPE.pkl', 'wb') as f:
        dump(all_bpe_ops, f)
