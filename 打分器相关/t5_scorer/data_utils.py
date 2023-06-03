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
    sents, labels, texts = [], [], []
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


def get_para_asqp_targets(labels, dataset):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            # at, ac, sp, ot = quad
            holder, target, polarity, expression = quad
            if 'mpqa' in dataset or 'ds' in dataset:
                polarity = sentword2opinion[polarity]  # 'POS' -> 'good'    

                one_quad_sentence = f"{holder} is {polarity} because {target} is {expression}" 
            else:
                one_quad_sentence = f""

            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences) 
        targets.append(target)
    return targets


def get_transformed_io(data_path, data_dir):
    """
    The main function to transform input & target according to the task
    """
    sents, labels, texts = read_line_examples_from_file(data_path, silence=True)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]

    targets = get_para_asqp_targets(labels, data_dir)

    return inputs, targets, texts


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'../data/{data_dir}/{data_type}.json'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir

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
        inputs, targets, sents = get_transformed_io(self.data_path, self.data_dir)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = "<suffix>" + targets[i]
            input = '<prefix>' + input
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
            self.sents = sents
