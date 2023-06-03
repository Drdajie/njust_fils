# -*- coding: utf-8 -*-
import json
# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
# sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}
sentword2opinion = {'positive': 'great', 'uncertain-positive': 'great', 
                     'negative': 'bad', 'uncertain-negative': 'bad',
                     'neutral': 'ok', 'both': 'ok', 'uncertain-neutral': 'ok', 'uncertain-both': 'ok'}

# def read_line_examples_from_file(data_path, dataset, silence):
def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels, texts = [], [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()
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
                sp = opinion['Polarity'].lower() if opinion['Polarity'] else 'NULL'
                ot = ' '.join(opinion['Polar_expression'][0]) if opinion['Polar_expression'][0] else 'NULL'
                quad_list.append([at, ac, sp, ot])
            labels.append(quad_list)
            texts.append(text)
            text = text.split()
            sents.append(text)
    if silence:
        print(f"Total examples = {len(sents)}")
        print("input example: ", sents[0])
        print("label example: ", labels[0])

    return sents, labels, texts


def get_para_asqp_inputs_targets(inputs, labels, dataset):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    multi_inputs, targets = [], []
    assert len(inputs) == len(labels)
    # inputs, labels = inputs[:20], labels[:20]
    for input, label in zip(inputs, labels):
        all_quad_sentences = []

        hs, ts, es, ps = [], [], [], []
     
        for quad in label:
            # at, ac, sp, ot = quad
            # print(quad)
            holder, aspect, polarity, expression = quad
            if 'mpqa' in dataset or 'ds' in dataset:
                polarity = sentword2opinion[polarity]
                one_quad_sentence = f"Holder refers to {holder}. " + \
                         f"Target refers to {aspect}. " + \
                         f"Expression refers to {expression}. " + \
                         f"Polarity refers to {polarity}."
            hs.append(holder)
            ts.append(aspect)
            es.append(expression)
            ps.append(polarity)
            all_quad_sentences.append(one_quad_sentence)
        target = ' [SSEP] '.join(all_quad_sentences)
        # print(target)
            
        if 'mpqa' in dataset or 'ds' in dataset:
            # 构造第一个
            one_pre = input
            one_set = []
            for h in hs:
                one_set.append(f"Holder refers to {h}.")
                # one_target += f"{h}, "
            # one_target = one_target[:-2] + '. '
            one_target = ' [SSEP] '.join(one_set)

            # 构造第二个
            two_pre = input + ' ' + one_target
            two_set = []
            for t in ts:
                two_set.append(f"Target refers to {t}.")
            two_target = ' [SSEP] '.join(two_set)
            for i in range(len(two_set)):
                two_set[i] = one_set[i] + ' ' + two_set[i]
            two_after = ' [SSEP] '.join(two_set)

            # 构造第三个
            three_pre = input + ' ' + two_after
            three_set = []
            for e in es:
                three_set.append(f"Expression refers to {e}.")
            three_target = ' [SSEP] '.join(three_set)
            for i in range(len(three_set)):
                three_set[i] = two_set[i] + ' ' + three_set[i]
            three_after = ' [SSEP] '.join(three_set)

            # 构造第四个
            four_pre = input + ' ' + three_after
            four_set = []
            for p in ps:
                four_set.append(f"Polarity refers to {p}.")
            four_target = ' [SSEP] '.join(four_set)


        multi_inputs.append(one_pre), targets.append(one_target)
        multi_inputs.append(two_pre), targets.append(two_target)
        multi_inputs.append(three_pre), targets.append(three_target)
        multi_inputs.append(four_pre), targets.append(four_target)

        # multi_inputs.append(input)
        # targets.append(target)

    return multi_inputs, targets


def get_test_targets(labels, dataset):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            # at, ac, sp, ot = quad
            holder, aspect, polarity, expression = quad
            if 'mpqa' in dataset or 'ds' in dataset:
                polarity = sentword2opinion[polarity]  # 'POS' -> 'good'    
                one_quad_sentence = f"Holder refers to {holder}. " + \
                                    f"Target refers to {aspect}. " + \
                                    f"Expression refers to {expression}. " + \
                                    f"Polarity refers to {polarity}."
            else:
                one_quad_sentence = f""
            all_quad_sentences.append(one_quad_sentence)
        target = ' [SSEP] '.join(all_quad_sentences) 
        targets.append(target)
    return targets


def get_transformed_io(data_path, data_dir, data_type):
    """
    The main function to transform input & target according to the task
    """
    sents, labels, texts = read_line_examples_from_file(data_path, silence=True)

    # the input is just the raw sentence

    if data_type == 'train':
        inputs, targets = get_para_asqp_inputs_targets(texts, labels, data_dir)
    else:
        inputs = [s for s in texts]
        targets = get_test_targets(labels, data_dir)

    print(len(inputs))
    print()
    print()
    print()

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'../data/{data_dir}/{data_type}.json'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.data_type = data_type

        self.inputs = []
        self.targets = []
        
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "text": self.sents[index]}

    def _build_examples(self):
        inputs, targets = get_transformed_io(self.data_path, self.data_dir, self.data_type)

        for i in range(len(inputs)):
            # change input and target to two strings
            # input = ' '.join(inputs[i])
            target = "<suffix>" + targets[i]
            input = '<prefix>' + inputs[i]
            target = targets[i]
            input = inputs[i]
            # if self.data_type == 'test':
                # print(input)
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
        self.sents = inputs