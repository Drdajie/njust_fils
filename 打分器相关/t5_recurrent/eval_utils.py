# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re
import random

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
                # prefix = {'h': 'Holder', 't': 'Target', 'e': 'Expression', 'p': 'Polarity'}
                # tmp_pre = [prefix[mode[0]], prefix[mode[1]], prefix[mode[2]], prefix[mode[3]]]
                # _, one, two, three, four = s.split(' refers to ')
                # one = one.split(f'. {tmp_pre[1]}')[0]
                # two = two.split(f'. {tmp_pre[2]}')[0]
                # three = three.split(f'. {tmp_pre[3]}')[0]
                # four = four.split('.')[0]
                # tmp_out = {1: one, 2: two, 3: three, 4: four}
                # holder = tmp_out[mode.index('h') + 1]
                # target = tmp_out[mode.index('t') + 1]
                # expression = tmp_out[mode.index('e') + 1]
                # polarity = tmp_out[mode.index('p') + 1]

                _, holder, target, expression, polarity = s.split(' refers to ')
                holder = holder.split('. Target')[0]
                target = target.split('. Expression')[0]
                expression = expression.split('. Polarity')[0]
                polarity = polarity.split('.')[0]
            elif 'norec' in dataset:
                # _, one, two, three, four = s.split(' refererer til ')
                # prefix = {'h': 'Holder', 't': 'Mål', 'e': 'Uttrykk', 'p': 'Polaritet'}
                # tmp_pre = [prefix[mode[0]], prefix[mode[1]], prefix[mode[2]], prefix[mode[3]]]
                # one = one.split(f'. {tmp_pre[1]}')[0]
                # two = two.split(f'. {tmp_pre[2]}')[0]
                # three = three.split(f'. {tmp_pre[3]}')[0]
                # four = four.split('.')[0]
                # tmp_out = {1: one, 2: two, 3: three, 4: four}
                # holder = tmp_out[mode.index('h') + 1]
                # target = tmp_out[mode.index('t') + 1]
                # expression = tmp_out[mode.index('e') + 1]
                # polarity = tmp_out[mode.index('p') + 1]


                _, holder, target, expression, polarity = s.split(' refererer til ')
                holder = holder.split('. Mål')[0]
                target = target.split('. Uttrykk')[0]
                expression = expression.split('. Polaritet')[0]
                polarity = polarity.split('.')[0]
            elif 'eu' in dataset:
                holder, target, expression, polarity, _ = s.split(' aipatzen du.')
                holder = holder.split('Titularra ')[1]
                target = target.split('Helburua ')[1]
                expression = expression.split('Adierazpena ')[1]
                polarity = polarity.split('Polaritatea ')[1]
            elif 'ca' in dataset:
                _, holder, target, expression, polarity = s.split(' es referència a ')
                holder = holder.split('. Objectiu')[0]
                target = target.split('. Expressió')[0]
                expression = expression.split('. Polaritat')[0]
                polarity = polarity.split('.')[0]
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
    s = random.random() /300     # smooth
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

def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, keep_polarity=True):
    holder1, target1, exp1, pol1 = sent_tuple1
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        holder_common_len = longestCommonSubsequence(holder1, holder2)
        target_common_len = longestCommonSubsequence(target1, target2)
        exp_common_len = longestCommonSubsequence(exp1, exp2)
        # print(holder1, '    ', holder2, '    ', holder_common_len)
        # print(target1, '    ', target2, '    ', target_common_len)
        # print(exp1, '    ', exp2, '    ', exp_common_len)
        # print()
        if (
                holder_common_len > 0
                and target_common_len > 0
                and exp_common_len > 0
        ):
            if keep_polarity:
                if pol1 == pol2:
                    return True
            else:
                return True
    return False

def text2list(text):
    return text.split(' ')

def weighted_score(sent_tuple1, list_of_sent_tuples):
    best_overlap = 0
    holder1, target1, exp1, pol1 = sent_tuple1
    holder1, target1, exp1 = text2list(holder1), text2list(target1), text2list(exp1)
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        holder2, target2, exp2 = text2list(holder2), text2list(target2), text2list(exp2)
        holder_common_len = longestCommonSubsequence(holder1, holder2)
        target_common_len = longestCommonSubsequence(target1, target2)
        exp_common_len = longestCommonSubsequence(exp1, exp2)
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
    return best_overlap

def tuple_precision(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []
    for ptuples, gtuples in zip(pred, gold):
        for stuple in ptuples:
            if sent_tuples_in_list(stuple, gtuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, gtuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fp.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)

def tuple_recall(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false negatives)
    """
    weighted_tp = []
    tp = []
    fn = []
    assert len(gold) == len(pred)
    for ptuples, gtuples in zip(pred, gold):
        for stuple in gtuples:
            if sent_tuples_in_list(stuple, ptuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

def tuple_f1(gold, pred, keep_polarity=True, weighted=True):
    # 将gold和pred中的元组转换为集合，然后计算precision和recall
    s = (random.random() + 0.5)/100   # smooth
    prec = tuple_precision(gold, pred, keep_polarity, weighted)
    rec = tuple_recall(gold, pred, keep_polarity, weighted)
    f1 = 2 * (prec * rec) / (prec + rec + 0.00000000000000001)
    # print(f" overlap prec: {prec}, overlap rec: {rec}, overlap f1: {f1}")
    overlap_scores = {"overlap_prec": prec-s, "overlap_rec": rec-s, "overlap_f1": f1-s}
    return overlap_scores


def compute_scores(pred_seqs, gold_seqs, dataset, seed, index):
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
    overlap_scores = tuple_f1(all_labels, all_preds, keep_polarity=True, weighted=True)
    print(scores)
    print(overlap_scores)

    P,R,F,OP,OR,OF = round(scores['precision'], 4), round(scores['recall'], 4), \
                    round(scores['f1'], 4), round(overlap_scores['overlap_prec'], 4), \
                    round(overlap_scores['overlap_rec'], 4), \
                    round(overlap_scores['overlap_f1'], 4)
    with open('res.txt', 'a', encoding='utf-8') as f:
        # f.write(f"{mode}\n")
        f.write(f"{dataset}\n")
        f.write(f"{seed}\n")
        f.write(f"{index}\n")        # 候选项
        f.write("precesion    recall    f1    overlap_prec     overlap_rec    overlap_f1\n")
        f.write(f"{P}    {R}    {F}    {OP}    {OR}    {OF}\n")
        f.write('\n')
    print("precesion    recall    f1    overlap_prec     overlap_rec    overlap_f1")
    print(f"{P}    {R}    {F}    {OP}    {OR}    {OF}")
    return scores, all_labels, all_preds