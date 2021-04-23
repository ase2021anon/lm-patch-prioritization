import xml.etree.ElementTree as ET
import os
import argparse
import tqdm
import re
from tree import DataASTNode

def ns_remover(tag):
    return re.sub(r'\{.*\}', '', tag)

def find_nodes(node, query):
    if ns_remover(node.tag) == query:
        return [node]
    else:
        return sum(map(lambda x: find_nodes(x, query), node.getchildren()), [])

def find_xml_files(dir_name):
    return [os.path.join(dir_name, x) 
            for x in os.listdir(dir_name) if '.java.xml' in x]

def list_join(joiner, org_list):
    if len(org_list) == 0:
        return org_list
    else:
        base_list = []
        for sublist in org_list[:-1]:
            base_list += sublist
            base_list += joiner
        base_list += org_list[-1]
        return base_list

def get_tokens(node, token_normalize=False, literal_normalize=True, whitelist=set()):
    def get_all_toks(node):
        if node.node_type == 'comment':
            return []

        all_child_tok_list = [get_all_toks(kid) for kid in node.children]
        all_child_toks = sum(all_child_tok_list, [])
        
        if len(node.token) == 0:
            my_tok = node.token
        elif ((node.node_type == 'name' and
               (token_normalize and node.token not in whitelist)) or
              ('literal' in node.node_type and
               (literal_normalize and node.token not in whitelist))):
            my_tok = '@' + node.node_type
        else:
            my_tok = node.token
        
        if node.tail is None:
            postfix = []
        else:
            postfix = [node.tail.strip()]
        return [my_tok] + all_child_toks + postfix

    pre_final_list = list(filter(lambda x: len(x) > 0, get_all_toks(node)))
    if len(node.tail.strip()) != 0:
        # don't want trailing stuff if it exists
        pre_final_list = pre_final_list[:-1]
    return pre_final_list
    

def XMLAST2procAST(root, vocab2idx=lambda x: x):
    new_root = DataASTNode(
        ns_remover(root.tag), 
        token=(root.text if root.text is not None else '').strip(),
        tail=(root.tail if root.tail is not None else '').strip()
    )
    new_root.add_index(vocab2idx(new_root.node_type))
    for child in root.getchildren():
        new_root.add_child(XMLAST2procAST(child, vocab2idx))
    return new_root

### Localizing functions ###

def get_patch_line(patch_file):
    '''Gets which line was modified from patch file. 
    assumes only one-hunk patches are passed to function.'''
    with open(patch_file) as f:
        for line in f:
            if line[:2] == '@@':
                loc_line = line
                break
        try:
            patch_line = int(loc_line.split()[2][1:].split(',')[0])+3
        except UnboundLocalError:
            print(f'File {patch_file} caused error;')
            return -1
    return patch_line

def pos_adder(name):
    return '{http://www.srcML.org/srcML/position}'+name

def get_node_pos(node):
    start_line = int(node.attrib[pos_adder('start')].split(':')[0])
    end_line = int(node.attrib[pos_adder('end')].split(':')[0])
    return start_line, end_line

def find_node_w_pos(pos, target, root=None, xml_file=None):
    def _find_ancestry_w_pos(pos, root):
        for child in root.getchildren():
            sl, el = get_node_pos(child)
            if sl == pos:
                return [root]
            elif sl < pos < el:
                return [root] + _find_ancestry_w_pos(pos, child)
            else:
                continue
        return []
    
    if root is None:
        if xml_file is None:
            raise ValueError('either root or xml_file must be provided')
        root = ET.parse(xml_file).getroot()
    node_path = _find_ancestry_w_pos(pos, root)
    target_node_list = [e for e in node_path if target in e.tag]
    return target_node_list[:1]

### Localizing functions over ###

def tokenize_files(file_list, params):
    f = open(params.target_file, 'w')
    for xml_file in tqdm.tqdm(file_list):
        print('//' + xml_file, file=f)

        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue
    
        if params.preprocessing == 1:
            target_nodes = find_nodes(tree.getroot(), params.target_node)
        else:
            try:
                patch_line = get_patch_line(xml_file.rstrip('.java.xml'))
                target_nodes = find_node_w_pos(patch_line, params.target_node, tree.getroot())
            except IndexError:
                print('Could not find patch line for file', xml_file)
                pass
            except FileNotFoundError:
                print('No patch exists for file', xml_file)
                continue
            except KeyError:
                print('Something wrong with position tags in file', xml_file)
                continue
            
            if len(target_nodes) == 0:
                print('error happened at %s, target line %s' % (xml_file, patch_line))
                # exit(0)
                continue

        for target_node in target_nodes:
            proc_node = XMLAST2procAST(target_node)
            tokens = get_tokens(
                proc_node,
                token_normalize=False,
                literal_normalize=True,
            )
            pending_str = ' '.join(tokens)
            if len(pending_str) > params.len_threshold or 'assert' in pending_str.lower():
                if params.preprocessing != 1:
                    # print(f'Length too long ({len(pending_str)}/{params.len_threshold}) or test ({"assert" in pending_str.lower()})')
                    pass
                continue

            try:
                pending_str.encode('ascii') # filters out non-ascii data
            except UnicodeEncodeError:
                if params.preprocessing != 1:
                    print('nonascii data detected in', xml_file)
                continue

            # write data to file
            print(pending_str, file=f)
        
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change XML files to tokens')
    parser.add_argument('--source_dir', help='Directory to find XML files')
    parser.add_argument('--target_node', type=str, default='function',
                        help='Which node becomes root of one line')
    parser.add_argument('--target_file', type=str, default='tokenized.txt',
                        help='Where to put tokenized results')
    parser.add_argument('--len_threshold', type=int, default=1000,
                        help='Maximum character length of sequence')
    parser.add_argument('--preprocessing', type=int, default=0,
                        help='If 1, will not look for patch files.')

    args = parser.parse_args()
    file_list = find_xml_files(args.source_dir)
    tokenize_files(file_list, args)
