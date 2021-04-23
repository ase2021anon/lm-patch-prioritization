import argparse
import tqdm
from pickle import load, dump

from bpe_utils import *

def cmd2BPEtok(cmd, bpe_ops, lower=True, cmd_level=False):
    '''Transforms cmd into BPE tokens.'''
    # some preprocessing
    cmd = cmd.strip()
    if lower:
        cmd = cmd.lower()
    tok_cmd = list(cmd)
    if not cmd_level:
        tok_cmd = [EOT if t == ' ' else t for t in tok_cmd]
        tok_cmd += [EOT]
    
    # actual merging
    subst_cmd = wrap_sep(TMP_SEP_TOK.join(tok_cmd))
    for bpe_idx in sorted(bpe_ops.keys()):
        bpe_pair = bpe_ops[bpe_idx]
        rep_str, new_str = wrap_sep(TMP_SEP_TOK.join(bpe_pair)), wrap_sep(''.join(bpe_pair))
        subst_cmd = subst_cmd.replace(rep_str, new_str)
        subst_cmd = subst_cmd.replace(rep_str, new_str) # overlap issues
    tok_cmd = unwrap_sep(subst_cmd).split(TMP_SEP_TOK)
    return tok_cmd

def parse_file(args):
    with open(args.tokenized_file, 'r') as f:
        cmd_lines = f.readlines()
    
    with open(args.BPE_ops_file, 'rb') as f:
        bpe_ops = load(f)
    
    all_newcmd_chars = []
    prev_name = 'PLACEHOLDER'
    prev_name_used = False
    for line in tqdm.tqdm(cmd_lines):
        if line[:2] == '//': 
            if (not prev_name_used) and (prev_name != 'PLACEHOLDER'):
                all_newcmd_chars.append((prev_name, []))
            prev_name = line[2:].strip()
            prev_name_used = False
            continue
        assert (not prev_name_used) or (prev_name == 'PLACEHOLDER')
        line_bpe_toks = cmd2BPEtok(line, bpe_ops, cmd_level=args.command_level)
        all_newcmd_chars.append((prev_name, line_bpe_toks))
        prev_name_used = True
    
    with open(args.target_file, 'wb') as f:
        dump(all_newcmd_chars, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse tokens into BPE tokens')
    parser.add_argument('--BPE_ops_file', help='File where BPE merge sequence is saved')
    parser.add_argument('--tokenized_file', help='File where tokenized code resides')
    parser.add_argument('--target_file', type=str, default='bpe_tokenized.pkl',
                        help='Where to save BPE-tokenized results')
    parser.add_argument('--command_level', type=int, default=0,
                        help='set to 1 if BPE was configured on sentence level, not token level.')
                        
    args = parser.parse_args()
    parse_file(args)
