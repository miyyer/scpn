
import torch, sys, h5py
import numpy as np
from nltk import ParentedTree
reload(sys)
sys.setdefaultencoding('utf8')

def is_paren(tok):
    return tok == ")" or tok == "("


# given list of parse strings, output numpy array containing the transformations
def indexify_transformations(in_p, out_p, label_voc, args):

    in_seqs = []
    out_seqs = []
    mismatch_inds = []

    max_trans_size = 0
    for idx in range(len(in_p)):

        # very rarely, a tree is invalid
        try:
            in_tree = ParentedTree.fromstring(in_p[idx])
            out_tree = ParentedTree.fromstring(out_p[idx])
        except:
            continue

        if args.tree_dropout > 0:
            tree_dropout(in_tree, args.tree_dropout, 0)
            tree_dropout(out_tree, args.tree_dropout, 0)
        elif args.tree_level_dropout > 0:
            parse_tree_level_dropout(in_tree, args.tree_level_dropout)
            parse_tree_level_dropout(out_tree, args.tree_level_dropout)

        in_full_trans = deleaf(in_tree)
        out_full_trans = deleaf(out_tree)

        if max_trans_size < len(in_full_trans):
            max_trans_size = len(in_full_trans)
        if max_trans_size < len(out_full_trans):
            max_trans_size = len(out_full_trans)

        # only consider instances where input syntax differs from output syntax
        if in_full_trans != out_full_trans:
            # make sure everything is invocab
            try:
                x = [label_voc[z] for z in in_full_trans]
                x = [label_voc[z] for z in out_full_trans]
                in_seqs.append(in_full_trans)
                out_seqs.append(out_full_trans)
                mismatch_inds.append(idx)
            except:
                pass

    # no syntactic transformations in the batch!
    if len(in_seqs) == 0:
        return None

    # otherwise, indexify and return
    else:
        in_trans_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')
        out_trans_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')

        in_lengths = []
        out_lengths = []
        for idx in range(len(in_seqs)):
            curr_in = in_seqs[idx]
            in_trans_np[idx, :len(curr_in)] = [label_voc[z] for z in curr_in]
            in_lengths.append(len(curr_in))

            curr_out = out_seqs[idx]
            out_trans_np[idx, :len(curr_out)] = [label_voc[z] for z in curr_out]
            out_lengths.append(len(curr_out))

        return in_trans_np, out_trans_np, mismatch_inds,\
            np.array(in_lengths, dtype='int32'), np.array(out_lengths, dtype='int32')


#returns tokenized parse tree and removes leaf nodes (i.e. words)
def deleaf(tree):
    nonleaves = ''
    for w in str(tree).replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split() + ['EOP']

#removes levels of parse tree belowe specifice level or random levels
#if level is None
def parse_tree_level_dropout(tree, treerate, level=None):
    def parse_tree_level_dropout2(tree, level, mlevel):
        if level == mlevel:
            for idx, n in enumerate(tree):
                if isinstance(n, ParentedTree):
                    tree[idx] = "(" + n.label() + ")"
        else:
            for n in tree:
                parse_tree_level_dropout2(n, level + 1, mlevel)

    h = tree.height()

    if not level:
        level = 0
        for i in range(2, h):
            if np.random.rand() <= treerate:
                level = i
                break
        if level > 0:
            parse_tree_level_dropout2(tree, 1, level)

    else:
        parse_tree_level_dropout2(tree, 1, level)

#dropout constituents from tree
def tree_dropout(tree, treerate, level):
    if level == 0:
        for n in tree:
            tree_dropout(n, treerate, level + 1)
    else:
        for idx, n in enumerate(tree):
            if np.random.rand(1)[0] <= treerate and isinstance(n, ParentedTree):
                tree[idx] = "(" + n.label() + ")"
            elif not isinstance(n, ParentedTree):
                continue
            else:
                tree_dropout(n, treerate, level + 1)

# given list of parse strings, output numpy array containing the transformations
def parse_indexify_transformations(in_p, out_p, label_voc, args):

    in_trimmed_seqs = []
    in_seqs = []
    out_trimmed_seqs = []
    out_seqs = []

    max_trans_size = 0
    for idx in range(len(in_p)):

        # very rarely, a tree is invalid
        try:
            in_trimmed = ParentedTree.fromstring(in_p[idx])
            in_orig = ParentedTree.fromstring(in_p[idx])
            out_trimmed = ParentedTree.fromstring(out_p[idx])
            out_orig = ParentedTree.fromstring(out_p[idx])
        except:
            continue

        out_dh = parse_tree_level_dropout(out_trimmed, args.tree_level_dropout)
        parse_tree_level_dropout(in_trimmed, args.tree_level_dropout, level=out_dh)

        in_orig = deleaf(in_orig)
        in_trimmed = deleaf(in_trimmed)
        out_orig = deleaf(out_orig)
        out_trimmed = deleaf(out_trimmed)

        if max_trans_size < len(in_orig):
            max_trans_size = len(in_orig)
        if max_trans_size < len(out_orig):
            max_trans_size = len(out_orig)

        # only consider instances where top-level of input parse != top-level output
        if in_trimmed != out_trimmed:
            # make sure everything is invocab
            try:             
                x = [label_voc[z] for z in in_orig]
                x = [label_voc[z] for z in out_orig]
                in_seqs.append(in_orig)
                out_seqs.append(out_orig)
                out_trimmed_seqs.append(out_trimmed)
                in_trimmed_seqs.append(in_trimmed)
            except:
                pass

    # no syntactic transformations in the batch!
    if len(in_seqs) == 0:
        return None

    # otherwise, indexify and return
    else:
        in_trans_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')
        out_trans_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')
        in_trimmed_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')
        out_trimmed_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')

        in_lengths = []
        out_lengths = []
        out_trimmed_lengths = []
        in_trimmed_lengths = []
        for idx in range(len(in_seqs)):
            curr_in = in_seqs[idx]
            in_trans_np[idx, :len(curr_in)] = [label_voc[z] for z in curr_in]
            in_lengths.append(len(curr_in))

            curr_out = out_seqs[idx]
            out_trans_np[idx, :len(curr_out)] = [label_voc[z] for z in curr_out]
            out_lengths.append(len(curr_out))

            curr_trimmed_in = in_trimmed_seqs[idx]
            in_trimmed_np[idx, :len(curr_trimmed_in)] = [label_voc[z] for z in curr_trimmed_in]
            in_trimmed_lengths.append(len(curr_trimmed_in))

            curr_trimmed_out = out_trimmed_seqs[idx]
            out_trimmed_np[idx, :len(curr_trimmed_out)] = [label_voc[z] for z in curr_trimmed_out]
            out_trimmed_lengths.append(len(curr_trimmed_out))

        # cut off extra padding
        in_trans_np = in_trans_np[:, :np.max(in_lengths)]
        out_trans_np = out_trans_np[:, :np.max(out_lengths)]
        in_trimmed_np = in_trimmed_np[:, :np.max(in_trimmed_lengths)]
        out_trimmed_np = out_trimmed_np[:, :np.max(out_trimmed_lengths)]

        return in_trans_np, out_trans_np, in_trimmed_np, out_trimmed_np,\
            np.array(in_lengths, dtype='int32'), np.array(out_lengths, dtype='int32'),\
            np.array(in_trimmed_lengths, dtype='int32'), np.array(out_trimmed_lengths, dtype='int32')

