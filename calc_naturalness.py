import numpy as np
from pickle import load
import os, sys
import argparse
import re
import tqdm

import torch
import torch.nn as nn
import javalang

from .models.model import LanguageModel
from .tokenizing.tokenize_xml import find_xml_files, find_node_w_pos, get_node_pos, get_patch_line

from .fl.read_fl_data import read_cov_data
from .fl.fl_methods import ochiai

DEVICE = 'cpu'
VOCAB_FILE = './bpe_lm/tokenizing/bpe_data/jm-trainfunc_vocab_BPE.pkl'
MODEL_PATH = './bpe_lm/models/weights/SrcMLBPE_JMJavaFunc_LMl1_z1000.pth'
Z_DIM = 1000
BIDIRECTIONAL = False

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab2idx = load(f)
    vocab2idx['@SOS'] = len(vocab2idx)
    vocab2idx['@EOS'] = len(vocab2idx)
    return vocab2idx

def cmd2longT(cmd, vocab2idx):
    cmd = ['@SOS'] + cmd + ['@EOS']
    idx_list = [vocab2idx[t]+1 for t in cmd if len(t) != 0]
    idx_tensor = torch.LongTensor(idx_list).unsqueeze(1).unsqueeze(1)
    return idx_tensor.to(DEVICE)

def load_model(model_path, vocab_size):
    lm = LanguageModel(vocab_size, hidden_size=Z_DIM, bidirectional=BIDIRECTIONAL)
    lm.load_state_dict(torch.load(model_path))
    lm.to(DEVICE)
    return lm

def eval_naturalness(cmd, lm, vocab2idx):
    loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
    longT = cmd2longT(cmd, vocab2idx)
    l_logits = lm(longT, [longT.size(0)-1])
    logits_flat = l_logits.view(-1, l_logits.size(-1))
    if BIDIRECTIONAL:
        labels_flat = longT[1:-1].view(-1)
    else:
        labels_flat = longT[1:].view(-1)
    naturalness = loss_fn(logits_flat, labels_flat)
    return naturalness.item()

def get_patch_reps(eval_file):
    with open(eval_file, 'rb') as f:
        eval_lines = load(f)
    name2bpe = {name:line for name, line in eval_lines}
    return name2bpe

def eval_over_file(eval_file, lm, vocab2idx):
    name2rep = get_patch_reps(eval_file)
    name2nat = dict()
    for name, line in tqdm.tqdm(name2rep.items()):
        if len(line) == 0:
            name2nat[name] = 99999.
        else:
            nat = eval_naturalness(line, lm, vocab2idx)
            name2nat[name] = nat
    return name2nat

def eval_suspiciousness(name2nat, proj, bug_num):
    (matrix, pass_data), _, line_names = read_cov_data(proj, bug_num)
    line2idx = {line: idx for idx, line in enumerate(line_names)}
    line_susp = ochiai(matrix, pass_data)

    name2susp = dict()
    for name in name2nat.keys():
        patch_file_name = name.strip('.java.xml')
        patch_line = get_patch_line(patch_file_name)
        with open(patch_file_name) as f:
            for line in f:
                if line[:3] == '---':
                    src_file_name = line.split()[1].strip()
                    src_file_dirs = src_file_name.split('/')
                    src_file_name = '/'.join(src_file_dirs[src_file_dirs.index('org'):])
                    src_file_name = src_file_name.strip('.java').replace('/', '.')
                    break

        local_lines = [f'{src_file_name}#{l}' for l in range(patch_line-1, patch_line+2)]
        local_susps = [line_susp[line2idx[line]]*(1+int(patch_line == line)) for line in local_lines
                       if line in line2idx]
        patch_susp = sum(local_susps) # no elements defaults to zero
        name2susp[name] = patch_susp
    return name2susp

def analyze_results(name2nat: dict):
    tup2name = dict()
    for n_idx, name in enumerate(name2nat.keys()):
        if 'Patch_' in name:
            patch_name = name[name.index('Patch_'):].rstrip('.txt.java.xml')
        else:
            faux_index = n_idx + 10**6
            patch_name = f'Buggy_{faux_index}_{faux_index}'
        total_idx, compilable_idx = list(map(lambda x: int(x) if x.isnumeric() else x, patch_name.split('_')[1:]))[:2]
        tup2name[(total_idx, compilable_idx)] = name
    sorted_patches = list(sorted(tup2name.keys()))

    name2res = dict()
    for order_idx, idxs in enumerate(sorted_patches[:-1]):
        (t_idx, c_idx) = idxs
        nt_idx, nc_idx = sorted_patches[order_idx+1]
        compilable = (c_idx != nc_idx or c_idx == 99999)
        plausible = (t_idx == nt_idx)
        name2res[tup2name[idxs]] = (compilable, plausible)
    last_name = tup2name[sorted_patches[-1]]
    name2res[last_name] = (False, False) # can't tell
    return name2res

def present_results(name2nat: dict, name2susp: dict, top_lines=10000, only_plausible=False):
    tup_results = [(nat_val/(name2susp[name]+1e-5), name) for name, nat_val in name2nat.items()]
    top_results = sorted(tup_results)
    name2res = analyze_results(name2nat)

    plausible_results = [0, 0, 0]
    compilable_results = [0, 0, 0]

    prev_idx = -1
    true_counter = 1
    for idx, (value, name) in zip(range(1, top_lines), top_results):
        if 'Patch_' in name:
            patch_name = name[name.index('Patch_'):].rstrip('.txt.java.xml')
        else:
            patch_name = 'Buggy_99999_99999'
        org_pidx = int(patch_name.split('_')[1])
        if org_pidx == prev_idx:
            continue
        compilable, plausible = name2res[name]
        rank_diff = (true_counter - org_pidx)
        res_idx = (int(rank_diff/abs(rank_diff)) if rank_diff != 0 else 0) + 1
        plausible_results[res_idx] += int(plausible)
        compilable_results[res_idx] += int(compilable)
        print(f'{true_counter} | {name} ({value:.4f})')# | {compilable} {plausible}')

        prev_idx = org_pidx
        true_counter += 1
    
    print('Plausible Patches Ranking Analysis (better, same, worse):', plausible_results)
    print('Compilable Patches:', compilable_results)

# Visualization code
def get_original_javafunc(patch_path, target_node='function'):
    patch_line = get_patch_line(patch_path)
    java_path = patch_path + '.java'
    xml_path = java_path + '.xml'
    
    func_node = find_node_w_pos(patch_line, target_node, xml_file=xml_path)[0]
    func_start, func_end = get_node_pos(func_node)
    
    with open(java_path) as f:
        java_lines = f.readlines()
        func_content = java_lines[func_start-1:func_end]
        no_comment_lines = []
        for jline in func_content:
            if '//' in jline:
                edit_jline = jline[:jline.index('//')] + '\n'
                if len(edit_jline.strip()) == 0:
                    continue
            else:
                edit_jline = jline
            no_comment_lines.append(edit_jline)
        indentation = len(func_content[0]) - len(func_content[0].lstrip())
        func_str = ''.join(e[min(indentation, len(e)-len(e.lstrip())):] for e in no_comment_lines)
        func_str = re.sub(re.compile(r"/\*([^*]|[\r\n])*?\*/", re.MULTILINE), "", func_str)
    return func_str

def prob2colorhex(p, low=(233,116,81), high=(144,238,144)):
    color_ints = [int(high[i]*p + low[i]*(1-p)) for i in range(3)]
    return '#'+''.join(hex(e)[2:] for e in color_ints)

def func_tokenize(raw_str):
    raw_tokens = [e.value for e in javalang.tokenizer.tokenize(raw_str)]
    true_tokens = []
    i = 0
    agglomerative_pairs = {
        ('(', ')'), ('[', ']'), ('return', ';'), ('continue', ';')
    }
    while i < len(raw_tokens):
        in_agg_pair = False
        for pair in agglomerative_pairs:
            if raw_tokens[i] == pair[0] and raw_tokens[i+1] == pair[1]:
                true_tokens.append(pair[0]+pair[1])
                i += 2
                in_agg_pair = True
        if not in_agg_pair:
            true_tokens.append(raw_tokens[i])
            i += 1
    return true_tokens

def get_html_rep(func_bpe, func_raw, vocab2idx, lm, name, top_k = 5):
    softmax_fn = nn.Softmax(dim=2)
    idx2vocab = {i+1:t for t, i in vocab2idx.items()}
    longT = cmd2longT(func_bpe, vocab2idx)
    l_logits = lm(longT, [longT.size(0)-1])
    probs = softmax_fn(l_logits)
    topk_probs, topk_tidxs = torch.topk(probs, k=top_k, dim=2)
    lm_favorites = [[(idx2vocab[top_idx.item()], topk_probs[tok_idx, 0, idx_idx].item())
                     for idx_idx, top_idx in enumerate(topk_tidxs[tok_idx, 0])]
                    for tok_idx in range(topk_tidxs.size(0))]
    tok_probs = [probs[idx, 0, ak.item()].item() 
                 for idx, ak in enumerate(longT[1:-1])]
    
    patch_html = ''
    curr_char_idx = 0
    curr_tok_idx = 0
    true_tokens = func_tokenize(func_raw)
    span_template = '<span class="gt" style="background-color:%s">%s<span class="pt">%s</span></span>'

    for bpe_tok, prob, favs in zip(func_bpe, tok_probs, lm_favorites):
        tok = bpe_tok.strip().replace('</t>', '')
        chomped_func_str = func_raw[curr_char_idx:]
        pre_ws_len = len(chomped_func_str) - len(chomped_func_str.lstrip())
        pre_ws = chomped_func_str[:pre_ws_len] # prewhitespace

        if (chomped_func_str[pre_ws_len:pre_ws_len+len(tok)].lower() == tok or
            tok == '@literal'):
            if tok == '@literal':
                next_part = true_tokens[curr_tok_idx]
            else:
                next_part = chomped_func_str[pre_ws_len:pre_ws_len+len(tok)]
            popup_str = '<br>'.join(f'{t}: {p:.2}' for t, p in favs)
            popup_str += f'<hr>{next_part}: {prob:.2f}'
            tok_html = (span_template % (prob2colorhex(prob), next_part, popup_str))
            patch_html += pre_ws + tok_html
            curr_char_idx += len(pre_ws + next_part)
            if '</t>' in bpe_tok:
                curr_tok_idx += 1
        else:
            print(f'Mismatch between \n{chomped_func_str} and \n{tok}, \n{true_tokens} \n{func_bpe}')
            patch_html = func_raw
            break
    
    full_html = f'''<button type="button" class="collapsible">{name}</button>
    <div class="content"><pre>{patch_html}</pre></div>'''
    return full_html

def hmean(a, b):
    return 2/(1/(a+1)+1/(b+1))

def get_visualization(name2bpe:dict, name2nat:dict, name2susp:dict, vocab2idx, lm, 
                      vis_shell='./bpe_lm/visualization_shell.html',
                      target_file='.html'):
    with open(vis_shell) as f:
        template_html = f.read()
        template_html = template_html.replace('%;', '%% ;')
    bug_name = 'd4j ' + list(name2nat.keys())[0].split('/')[-2]
    tup_results = [(value, name) for name, value in name2nat.items()]
    top_results = sorted(tup_results)
    name2lm_rank = {name: idx for idx, (_, name) in enumerate(top_results)}
    name2org_rank = {name: idx for idx, name in enumerate(sorted(name2nat.keys(), key=lambda x: int(re.findall(r'\d+', x.split('/')[-1])[0])))}
    tup_results = [(hmean(idx, name2org_rank[name]), name) for idx, (_, name) in enumerate(top_results)]
    top_results = sorted(tup_results)
    name2res = analyze_results(name2nat)
    htmls = []
    for idx, (value, name) in enumerate(top_results):
        bpe_rep = name2bpe[name]
        susp = name2susp[name]
        nat_val = name2nat[name]
        patch_name = name.split('/')[-1]
        compilable, plausible = name2res[name]
        cnp_str = f' [rank={idx+1}] [lm_rank={name2lm_rank[name]+1}] [compilable={compilable}] [plausible={plausible}] [nat={nat_val:.2f}] [susp={susp:.2f}]'
        raw_rep = get_original_javafunc(name.rstrip('.java.xml'))
        result_html = get_html_rep(bpe_rep, raw_rep, vocab2idx, lm, patch_name + cnp_str)
        htmls.append(result_html)
    all_patch_html = '\n'.join(htmls)
    final_html = template_html % (bug_name, all_patch_html)
    with open(target_file, 'w') as f:
        f.write(final_html)

def save_to_csv(csv_path, name2nat, name2susp):
    with open(csv_path, 'w') as f:
        print(f'PatchName,nat,susp', file=f)
        for name in name2nat:
            print(f'{name},{name2nat[name]},{name2susp[name]}', file=f)
    
if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description='Calculate naturalness of patches')
    parser.add_argument('--eval_file', help='Pickled file of BPE tokens')
    parser.add_argument('--top_n', type=int, default=10000,
                        help='Show top n natural patches and their names')
    parser.add_argument('--ref_dir', default=None, help="where to get xml file names")
    parser.add_argument('--only_plausible', type=int, default=0,
                        help='Show only plausible patches, based on total-idx overlap')
    parser.add_argument('--visualization_file', type=str, default='test.html',
                        help='Where to save visualization.')
    parser.add_argument('--use_suspiciousness', type=int, default=0,
                        help='Use SBFL suspiciousness.')
    parser.add_argument('--bug_proj', type=str, default='Chart',
                        help='(Used when use_suspiciousness=1) Bug project')
    parser.add_argument('--bug_num', type=int, default=1,
                        help='(Used when use_suspiciousness=1) Bug number')
    parser.add_argument('--model_path', type=str,
                        help='specify language model to use (might have to tweak with hyperparams in this file)')
    parser.add_argument('--finetuned', type=int, default=0,
                        help='whether to use fine-tuned models')
    parser.add_argument('--nat_csv', type=str,
                        help='where to save calculated naturalness information.')

    args = parser.parse_args()
    torch.no_grad()

    vocab2idx = load_vocab(VOCAB_FILE)
    vocab_size = (max(vocab2idx.values())+1)+1
    if args.finetuned:
        base = args.model_path.rstrip('.pth')
        used_model_path = base + '_' + args.bug_proj + 'Overfit.pth'
        print('using finetuned', used_model_path)
    else:
        used_model_path = args.model_path
    lm = load_model(used_model_path, vocab_size)

    if args.ref_dir is not None:
        names = find_xml_files(args.ref_dir)
    else:
        names = None

    name2nat = eval_over_file(args.eval_file, lm, vocab2idx)
    if args.use_suspiciousness:
        name2susp = eval_suspiciousness(name2nat, args.bug_proj, args.bug_num)
    else:
        name2susp = {name: 1. for name in name2nat.keys()}
    save_to_csv(args.nat_csv, name2nat, name2susp)

    name2bpe = get_patch_reps(args.eval_file)
    get_visualization(name2bpe, name2nat, name2susp, vocab2idx, lm, target_file=args.visualization_file)
