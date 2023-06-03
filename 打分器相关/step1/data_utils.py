# -*- coding: utf-8 -*-
import json
import random

from torch.utils.data import Dataset

# This script contains all data transformation and reading

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'uncertain-positive': 'great', 
                     'negative': 'bad', 'uncertain-negative': 'bad',
                     'neutral': 'ok', 'both': 'ok', 
                     'uncertain-neutral': 'ok', 'uncertain-both': 'ok'}
nor_p2o = {'positive': 'flink', 'uncertain-positive': 'flink', 
            'negative': 'dårlig', 'uncertain-negative': 'dårlig',
            'neutral': 'ok', 'both': 'ok', 'uncertain-neutral': 'ok', 'uncertain-both': 'ok'}
eu_p2o = {'positive': 'handia', 'uncertain-positive': 'handia', 
            'negative': 'txarra', 'uncertain-negative': 'txarra',
            'neutral': 'Ados', 'both': 'Ados', 'uncertain-neutral': 'Ados', 'uncertain-both': 'Ados'}
ca_p2o = {'positive': 'positiu', 'uncertain-positive': 'positiu', 
            'negative': 'dolent', 'uncertain-negative': 'dolent',
            'neutral': "D'acord", 'both': "D'acord", 'uncertain-neutral': "D'acord", 'uncertain-both': "D'acord"}


def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    num = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()
        # raw_data = raw_data[0]
        raw_data = json.loads(raw_data)

        for line in raw_data:
            sent_id, text, opinions = line['sent_id'], line['text'], line['opinions']
            if len(opinions) == 0:
                continue
            quad_list = []
            for opinion in opinions:
                if not opinion['Polarity'] or len(opinion['Source']) == 0:
                    continue
                at = ' '.join(opinion['Source'][0]) if opinion['Source'][0] else 'NULL'
                ac = ' '.join(opinion['Target'][0]) if opinion['Target'][0] else 'NULL'
                # sp = opinion['Polarity'].lower()
                sp = opinion['Polarity'].lower() if opinion['Polarity'] else 'NULL'
                ot = ' '.join(opinion['Polar_expression'][0]) if opinion['Polar_expression'][0] else 'NULL'
                # ot = ' '.join(opinion['Polar_expression'][0])
                if ot not in text:
                    continue
                quad_list.append([at, ac, sp, ot])
            if quad_list != []:
                num += 1
            labels.append(quad_list)

            text = text.split()
            sents.append(text)
    if silence:
        print(f"Total examples = {num}")
        print("input example: ", sents[0])
        print("label example: ", labels[0])

    return sents, labels


def get_para_asqp_targets(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    filted_sents = []
    for i, label in enumerate(labels):
        all_quad_sentences = []
        NP1S, NP2S, NP3S = [], [], []
        # print(label)
        # quad = label[0] if len(label) > 1 else label[0]
        if len(label) == 0:
            continue

        filted_sents.append(sents[i])

        for holder, target, polarity, expression in label:
            polarity = sentword2opinion[polarity]
            one_quad_sentence = f"Holder refers to {holder}. Target refers to {target}. Expression refers to {expression}. " \
                            f"Polarity refers to {polarity}."

            # 基于holder, target, man_ot, expression进行数据增强
            # 1.交换 Holder 和 Target，其余部分不变：
            NP1 = f"Holder refers to {target}. Target refers to {holder}. " \
                f"Expression refers to {expression}. Polarity refers to {polarity}."
            # 2.随机选择非 Expression 部分的与 Expression 相同词数的 span 作为 Expression，其余部分不变：（例子中 Expression 包含三个词，所以随机选择连续的其余部分的三个词作为 Expression）
            # 计算 Expression 的长度
            expression_len = len(expression.split())
            # 随机选择一个位置
            # random_idx = random.randint(0, len(one_quad_sentence.split(' ')) - expression_len)
            # print(len(sents[i]), expression_len)
            # if(expression_len > len(sents[i])):
            # print(expression, sents[i])
            random_idx = random.randint(0, len(sents[i]) - expression_len)
            # 选择连续的 expression_len 个词作为 Expression
            random_expression = ' '.join(sents[i][random_idx:random_idx + expression_len])
            # 生成新的句子
            NP2 = f"Holder refers to {holder}. Target refers to {target}. " \
                f"Expression refers to {random_expression}. Polarity refers to {polarity}. "

            # 3.选取 label 中不存在的 Polarity 作为 Polarity，其余不变：
            # 随机选择一个不存在的 Polarity, 例如，如果原始的 Polarity 为 POS，那么随机选择 NEG 或 NEU
            if polarity == 'great':
                random_polarity = random.choice(['bad', 'ok'])
            elif polarity == 'bad':
                random_polarity = random.choice(['great', 'ok'])
            elif polarity == 'ok':
                random_polarity = random.choice(['great', 'bad'])
            # 生成新的句子
            NP3 = f"Holder refers to {holder}. Target refers to {target}. " \
                f"Expression refers to {expression}. Polarity refers to {random_polarity}."
            

            all_quad_sentences.append(one_quad_sentence)
            NP1S.append(NP1)
            NP2S.append(NP2)
            NP3S.append(NP3)
        target = ' [SSEP] '.join(all_quad_sentences)
        ERROR1 = ' [SSEP] '.join(NP1S)
        ERROR2 = ' [SSEP] '.join(NP2S)
        ERROR3 = ' [SSEP] '.join(NP3S)
        # example = [one_quad_sentence, NP1, NP2, NP3]  # 生成四个句子，一个是原始的句子，三个是数据增强的负例句子
        # if mask == 'holder_target':
        #     example = [target, ERROR2, ERROR3]
        # elif mask == 'expression':
        #     example = [target, ERROR1, ERROR3]
        # elif mask == 'polarity':
        #     example = [target, ERROR1, ERROR2]
        # else:
        example = [target, ERROR1, ERROR2, ERROR3]

        # target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(example)
    return filted_sents, targets


def get_transformed_io(data_path, data_dir):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, silence=True)
    filted_sents, targets = get_para_asqp_targets(sents, labels)

    return filted_sents, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128, ori=False):
        # './data/rest16/train.txt'
        self.data_path = f'../data/{data_dir}/{data_type}.json'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        self.inputs = []
        self.targets = []
        self.ori = ori

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = [self.targets[index][i]["input_ids"].squeeze() for i in range(len(self.targets[index]))]

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = [self.targets[index][i]["attention_mask"].squeeze() for i in range(len(self.targets[index]))]
        item = {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }
        return item

    def _build_examples(self):
        inputs, targets = get_transformed_io(self.data_path, self.data_dir)
        try:
            assert len(inputs) == len(targets)
            print(f"Number of examples: {len(inputs)}")
        except AssertionError:
            print(f"Number of inputs: {len(inputs)}")
            print(f"Number of targets: {len(targets)}")
            raise AssertionError
            exit(1)
        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            # target = "<suffix>" + targets[i]
            # suffixs = ['<suffix>' + s for s in targets[i]]
            suffixs = [s for s in targets[i]]
            # input = '<prefix>' + input
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_targets = [
                self.tokenizer.batch_encode_plus(
                    [target], max_length=self.max_len, padding="max_length",
                    truncation=True, return_tensors="pt") for target in suffixs]
            # tokenized_target = self.tokenizer.batch_encode_plus(
            #     [target], max_length=self.max_len, padding="max_length",
            #     truncation=True, return_tensors="pt"
            # )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_targets)














# elif 'norec' in dataset:
#                 polarity = nor_p2o[polarity]
#                 one_quad_sentence = f"Holder refererer til {holder}. " + \
#                                     f"Mål refererer til {target}. " + \
#                                     f"Uttrykk refererer til {expression}. " + \
#                                     f"Polaritet refererer til {polarity}. "
                
#                 # 基于holder, target, man_ot, expression进行数据增强
#                 # 1.交换 Holder 和 Target，其余部分不变：
#                 NP1 = f"Holder refererer til {target}. Target refererer til {holder}. " \
#                     f"Expression refererer til {expression}. Polarity refererer til {polarity}."
#                 # 2.随机选择非 Expression 部分的与 Expression 相同词数的 span 作为 Expression，其余部分不变：（例子中 Expression 包含三个词，所以随机选择连续的其余部分的三个词作为 Expression）
#                 # 计算 Expression 的长度
#                 expression_len = len(expression.split())
#                 # 随机选择一个位置
#                 random_idx = random.randint(0, len(one_quad_sentence.split(' ')) - 3 - expression_len)
#                 # 选择连续的 expression_len 个词作为 Expression
#                 random_expression = ' '.join(one_quad_sentence.split(' ')[random_idx:random_idx + expression_len])
#                 # 生成新的句子
#                 NP2 = f"Holder refererer til {holder}. Target refererer til {target}. " \
#                     f"Expression refererer til {random_expression}. Polarity refererer til {polarity}. "

#                 # 3.选取 label 中不存在的 Polarity 作为 Polarity，其余不变：
#                 # 随机选择一个不存在的 Polarity, 例如，如果原始的 Polarity 为 POS，那么随机选择 NEG 或 NEU
#                 if polarity == 'flink':
#                     random_polarity = 'dårlig'
#                 elif polarity == 'dårlig':
#                     random_polarity = 'flink'
#                 # 生成新的句子
#                 NP3 = f"Holder refererer til {holder}. Target refererer til {target}. " \
#                     f"Expression refererer til {expression}. Polarity refererer til {random_polarity}."

#             elif 'eu' in dataset:
#                 polarity = eu_p2o[polarity]
#                 one_quad_sentence = f"Titularra {holder} aipatzen du. " + \
#                                     f"Helburua {target} aipatzen du. " + \
#                                     f"Adierazpena {expression} aipatzen du. " + \
#                                     f"Polaritatea {polarity} aipatzen du. "
#             elif 'ca' in dataset:
#                 polarity = ca_p2o[polarity]
#                 one_quad_sentence = f"Titular es referència a {holder}. " + \
#                                     f"Objectiu es referència a {target}. " + \
#                                     f"Expressió es referència a {expression}. " + \
#                                     f"Polaritat es referència a {polarity}. "
#                 # 基于holder, target, man_ot, expression进行数据增强
#                 # 1.交换 Holder 和 Target，其余部分不变：
#                 NP1 = f"Titular es referència a {target}. Objectiu es referència a {holder}. " \
#                     f"Expressió es referència a {expression}. Polaritat es referència a {polarity}."
#                 # 2.随机选择非 Expression 部分的与 Expression 相同词数的 span 作为 Expression，其余部分不变：（例子中 Expression 包含三个词，所以随机选择连续的其余部分的三个词作为 Expression）
#                 # 计算 Expression 的长度
#                 expression_len = len(expression.split())
#                 # 随机选择一个位置
#                 random_idx = random.randint(0, len(one_quad_sentence.split(' ')) - 3 - expression_len)
#                 # 选择连续的 expression_len 个词作为 Expression
#                 random_expression = ' '.join(one_quad_sentence.split(' ')[random_idx:random_idx + expression_len])
#                 # 生成新的句子
#                 NP2 = f"Titular es referència a {holder}. Objectiu es referència a {target}. " \
#                     f"Expressió es referència a {random_expression}. Polaritat es referència a {polarity}. "

#                 # 3.选取 label 中不存在的 Polarity 作为 Polarity，其余不变：
#                 # 随机选择一个不存在的 Polarity, 例如，如果原始的 Polarity 为 POS，那么随机选择 NEG 或 NEU
#                 if polarity == 'positiu':
#                     random_polarity = 'dolent'
#                 elif polarity == 'dolent':
#                     random_polarity = 'positiu'
#                 # 生成新的句子
#                 NP3 = f"Holder es referència a {holder}. Target es referència a {target}. " \
#                     f"Expression es referència a {expression}. Polarity es referència a {random_polarity}."
