# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}
numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def extract_spans_para(seq, dataset):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        # food quality is bad because pizza is over cooked.
        try:
            # ac, at, sp, ot 代表 holder、aspect、polarity、opinion

            if 'mpqa' in dataset or 'ds' in dataset:
                holder_polarity, target_expression = s.split(' because ')
                holder, polarity = holder_polarity.split(' is ')
                target, expression = target_expression.split(' is ')
            ac, at, ot, sp = holder, target, expression, polarity

        except ValueError:
            try:
                pass
            except UnicodeEncodeError:
                pass
            ac, at, sp, ot = '', '', '', ''
        except IndexError:
            ac, at, sp, ot = '', '', '', ''

        quads.append((ac, at, ot, sp))
    return quads


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for j, t in enumerate(pred_pt[i]):
            if j >= len(gold_pt[i]):
                break
            # print(f"TP: {t} , {gold_pt[i]}")
            if t in gold_pt[i]:
                n_tp += 1

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}
    # print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    return scores

def longestCommonSubsequence(words1, words2):
    """ 最长公共子序列
    words1、words2：对比句子（可以是 string 也可以是 list）
    """
    if type(words1) == str:
        words1, words2 = words1.split(' '), words2.split(' ')
    M, N = len(words1), len(words2)
    # M, N = len(text1), len(text2)
    dp = [[0] * (N + 1) for _ in range(M + 1)]
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[M][N]

def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, eval_mode):
    """
    eval_mode: holder, target, expression, nf1, f1
    """
    holder1, target1, exp1, pol1 = sent_tuple1

    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        holder_common_len = longestCommonSubsequence(holder1, holder2)
        target_common_len = longestCommonSubsequence(target1, target2)
        exp_common_len = longestCommonSubsequence(exp1, exp2)

        if eval_mode == 'holder':
            if holder_common_len > 0:   return True
        elif eval_mode == 'target':
            if target_common_len > 0:   return True
        elif eval_mode == 'expression':
            if exp_common_len > 0:  return True
        elif eval_mode == 'nf1':
            if (
                    holder_common_len > 0
                    and target_common_len > 0
                    and exp_common_len > 0
            ):  return True
        elif eval_mode == 'f1':
            if (
                    holder_common_len > 0
                    and target_common_len > 0
                    and exp_common_len > 0
                    and pol1 == pol2
            ): return True
    return False

def text2list(text):
    return text.split(' ')

def weighted_score(sent_tuple1, list_of_sent_tuples, eval_mode):
    best_overlap = 0
    longest_len = 0
    holder1, target1, exp1, pol1 = sent_tuple1
    holder1, target1, exp1 = text2list(holder1), text2list(target1), text2list(exp1)
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        holder2, target2, exp2 = text2list(holder2), text2list(target2), text2list(exp2)
        holder_common_len = longestCommonSubsequence(holder1, holder2)
        target_common_len = longestCommonSubsequence(target1, target2)
        exp_common_len = longestCommonSubsequence(exp1, exp2)

        if eval_mode == 'holder':
            if holder_common_len > longest_len:
                longest_len = holder_common_len
        elif eval_mode == 'target':
            if target_common_len > longest_len:
                longest_len = target_common_len
        elif eval_mode == 'expression':
            if exp_common_len > longest_len:
                longest_len = exp_common_len
        elif eval_mode == 'nf1' or eval_mode == 'f1':
            if (
                    holder_common_len > 0
                    and target_common_len > 0
                    and exp_common_len > 0
            ):  
                holder_overlap = holder_common_len / len(holder1)
                target_overlap = target_common_len / len(target1)
                exp_overlap = exp_common_len / len(exp1)
                overlap = (holder_overlap + target_overlap + exp_overlap) / 3
                if overlap > best_overlap:
                    best_overlap = overlap

    if eval_mode == 'holder':
        return longest_len, len(holder1)
    elif eval_mode == 'target':
        return longest_len, len(target1)
    elif eval_mode == 'expression':
        return longest_len, len(exp1)
    elif eval_mode == 'nf1' or eval_mode == 'f1':
        return best_overlap

def tuple_precision(gold, pred, eval_mode):
    """
    Weighted true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []
    assert len(gold) == len(pred)

    if eval_mode == 'holder' or eval_mode == 'target' \
        or eval_mode == 'expression':
        num_down, num_up = 0, 0
        for ptuples, gtuples in zip(pred, gold):
            for stuple in ptuples:
                tmp_up, tmp_down = weighted_score(stuple, gtuples, eval_mode)
                num_down += tmp_down
                num_up += tmp_up
        print(num_up, '   ',  num_down)
        return num_up / num_down
    elif eval_mode == 'nf1' or eval_mode == 'f1':
        for ptuples, gtuples in zip(pred, gold):
            for stuple in ptuples:
                if sent_tuples_in_list(stuple, gtuples, eval_mode):
                    weighted_tp.append(weighted_score(stuple, gtuples, eval_mode))
                    tp.append(1)
                else:   fp.append(1)

        return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)

def tuple_recall(gold, pred, eval_mode):
    """
    Weighted true positives / (true positives + false negatives)
    """
    weighted_tp = []
    tp = []
    fn = []
    assert len(gold) == len(pred)

    if eval_mode == 'holder' or eval_mode == 'target' \
        or eval_mode == 'expression':
        num_down, num_up = 0, 0
        for ptuples, gtuples in zip(pred, gold):
            for stuple in gtuples:
                tmp_up, tmp_down = weighted_score(stuple, ptuples, eval_mode)
                num_down += tmp_down
                num_up += tmp_up
        return num_up / num_down
    
    elif eval_mode == 'nf1' or eval_mode == 'f1':
        for ptuples, gtuples in zip(pred, gold):
            for stuple in gtuples:
                if sent_tuples_in_list(stuple, ptuples, eval_mode):
                    weighted_tp.append(weighted_score(stuple, ptuples, eval_mode))
                    tp.append(1)
                else:   fn.append(1)
        return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

def tuple_f1(gold, pred):
    # span holder
    h_prec = tuple_precision(gold, pred, 'holder')
    h_rec = tuple_recall(gold, pred, 'holder')
    h_f1 = 2 * (h_prec * h_rec) / (h_prec + h_rec + 0.00000000000000001)
    # span target
    t_prec = tuple_precision(gold, pred, 'target')
    t_rec = tuple_recall(gold, pred, 'target')
    t_f1 = 2 * (t_prec * t_rec) / (t_prec + t_rec + 0.00000000000000001)
    # span expression
    e_prec = tuple_precision(gold, pred, 'expression')
    e_rec = tuple_recall(gold, pred, 'expression')
    e_f1 = 2 * (e_prec * e_rec) / (e_prec + e_rec + 0.00000000000000001)
    # nf1
    n_prec = tuple_precision(gold, pred, 'nf1')
    n_rec = tuple_recall(gold, pred, 'nf1')
    n_f1 = 2 * (n_prec * n_rec) / (n_prec + n_rec + 0.00000000000000001)
    # f1
    f_prec = tuple_precision(gold, pred, 'f1')
    f_rec = tuple_recall(gold, pred, 'f1')
    f_f1 = 2 * (f_prec * f_rec) / (f_prec + f_rec + 0.00000000000000001)
    # print(f" overlap prec: {prec}, overlap rec: {rec}, overlap f1: {f1}")
    overlap_holder_scores = {'op': h_prec, 'or': h_rec, 'of1': h_f1}
    overlap_target_scores = {'op': t_prec, 'or': t_rec, 'of1': t_f1}
    overlap_exp_scores = {'op': e_prec, 'or': e_rec, 'of1': e_f1}
    overlap_nf1_scores = {'op': n_prec, 'or': n_rec, 'of1': n_f1}
    overlap_f1_scores = {'op': f_prec, 'or': f_rec, 'of1': f_f1}
    return overlap_holder_scores, overlap_target_scores, overlap_exp_scores, overlap_nf1_scores, overlap_f1_scores


def compute_scores(pred_seqs, gold_seqs, dataset):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(gold_seqs[i], dataset)
        pred_list = extract_spans_para(pred_seqs[i], dataset)

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)
    over_scores = tuple_f1(all_labels, all_preds) #, keep_polarity=True, weighted=True)
    for i in range(len(over_scores)):
        over_scores[i]['op'] = round(over_scores[i]['op'], 4)
        over_scores[i]['or'] = round(over_scores[i]['or'], 4)
        over_scores[i]['of1'] = round(over_scores[i]['of1'], 4)

    P,R,F = round(scores['precision'], 4), round(scores['recall'], 4), \
                    round(scores['f1'], 4)
    with open('res.txt', 'a', encoding='utf-8') as f:
        # f.write(f"{mode}\n")
        f.write(f"{dataset}\n")
        names = ['holder', 'target', 'expression', 'nf1', 'f1']
        f.write("precesion    recall    f1\n")
        f.write(f"{P}           {R}             {F}\n")
        for name in names:
            f.write(f"{name}    ")
        f.write('\n')
        for i, name in enumerate(names):
            f.write(f"{over_scores[i]['of1']}   ")
        f.write('\n\n')
    print("precesion    recall    f1    overlap_prec     overlap_rec    overlap_f1")
    print(f"{P}    {R}    {F}    {over_scores[-1]['op']}    {over_scores[-1]['or']}    {over_scores[-1]['of1']}")
    return scores, all_labels, all_preds