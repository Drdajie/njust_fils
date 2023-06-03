# -*- coding: utf-8 -*-

import argparse
import json
import os
import logging
import time
import pickle

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, Adafactor, AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from data_utils import ABSADataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores

# RankGen
from transformers import AutoModelForMaskedLM
import torch.nn as nn



logger = logging.getLogger(__name__)


# print(torch.cuda.is_available())
# print(torch.cuda.device_count())


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='darmstadt_unis', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true',
                        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--hops', type=int, default=4)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=1024,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    # extra settings
    parser.add_argument("--ori", action='store_true', help="Whether to use the original data")

    # 额外增加的rankgen的参数
    parser.add_argument('--retriever_model_path', default='./t5-base', type=str)
    parser.add_argument('--cache_dir', default="./rankgen_cache_dir", type=str)
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--beam_size', default=6, type=int)
    parser.add_argument('--num_tokens', default=20, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--f_mode', type=bool, default=True)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # RankGen
    parser.add_argument('--pretrain_model_path', type=str)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3,
                        help="hidden dropout probability")
    parser.add_argument('--hidden_size', type=int, default=768,
                        help="hidden size")

    parser.add_argument('--index', type=int)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if args.f_mode == True:
        args.hops = 1
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir
    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                       data_type=type_path, max_len=args.max_seq_length)


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """

    def __init__(self, args, tfm_model, tokenizer, vocab_size=32130):
        super(T5FineTuner, self).__init__()
        self.args = args
        self.model = tfm_model

        self.tokenizer = tokenizer

        # self.model.encoder.resize_token_embeddings(len(tokenizer))

        # 构建rankgen的encoder和decoder

        self.total_steps = 0
        self.total_rank_loss = 0
        self.total_t5_loss = 0
        self.total_loss = 0

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        self.total_steps += 1
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return outputs

    def rank_loss(self, logits, labels):
        zpi = torch.sum(torch.exp(torch.matmul(logits, labels.t()).squeeze(dim=0)), dim=-1)
        p_ci_pi = torch.cat(
            [torch.exp(torch.matmul(logits[i].view(1, -1), labels[i].view(-1, 1)).squeeze(dim=0)) for i in
             range(logits.shape[0])],
            dim=0) / zpi
        loss = -torch.sum(torch.log(p_ci_pi)) / logits.shape[0]
        return loss

    def _step(self, batch):
        # 训练生成模型
        lm_labels = batch["target_ids"].clone()
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]
        if self.args.debug:
            os.system('nvidia-smi')
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_rankgen_loss", rankgen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def _training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        print(
            f"avg rankgen loss: {self.total_rank_loss / self.total_steps :.4f}\t "
            f"avg t5 loss: {self.total_t5_loss / self.total_steps :.4f}\t "
            f"avg total loss: {self.total_loss / self.total_steps :.4f}")
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def _validation_step(self, batch, batch_idx):
        loss, rankgen_loss = self._step(batch)
        return {"val_loss": loss, "val_rank_loss": rankgen_loss}

    def _validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def _optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # pass
        # if self.trainer.use_tpu:
        #     xm.optimizer_step(optimizer)
        # else:
        #     optimizer.step()
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.args)
        dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.args.train_batch_size * max(1, len(self.args.n_gpu))))
                // self.args.gradient_accumulation_steps
                * float(self.args.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.args)
        return DataLoader(val_dataset, batch_size=self.args.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


class RankEncoder(nn.Module):
    def __init__(self, config):
        super(RankEncoder, self).__init__()
        self.encoder = AutoModelForMaskedLM.from_pretrained('/root/data3/bert-base-uncased').bert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        input_ids = input_ids.cuda()
        if attention_mask != None:
            attention_mask = attention_mask.cuda()
        if token_type_ids != None:
            token_type_ids = token_type_ids.cuda()
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # print(outputs)
        # exit(0)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def evaluate(data_loader, model, args):
    """
    Compute scores given the predictions and gold labels
    """
    dataset, seed = args.dataset, args.seed
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.model.to(device)
    else:
        device = torch.device('cpu')
        model.model.to(device)
    model.model.to(device)

    model.model.eval()

    outputs, targets = [], []
    all_outputs = []
    texts = []

    scorer_tokenizer = AutoTokenizer.from_pretrained('/root/data3/bert-base-uncased')
    scorer_tokenizer.add_special_tokens({'additional_special_tokens': ['<prefix>', '<suffix>']})
    scorer_model = RankEncoder(args)
    scorer_model.encoder.resize_token_embeddings(len(scorer_tokenizer))   # resize the token embeddings
    scorer_model.load_state_dict(torch.load(f'../step1/outputs/{dataset}/model.pth'))
    scorer_model.cuda()
    scorer_model.eval()

    for batch in tqdm(data_loader):
        pre = batch['source_ids']       # 输入的前缀
        # print(batch['text'])
        pre_set = []                    # ['Holder refers to A.', 'Holder refers to B.']
        for i in range(1):
            curr_outs = model.model.generate(input_ids=pre.to(device),
                                            attention_mask=batch['source_mask'].to(device),
                                            max_length=128,
                                            return_dict_in_generate=True,
                                            num_return_sequences=8,
                                            num_beams=8)  # num_beams=8, early_stopping=True)
            curr_outs_text = model.tokenizer.batch_decode(curr_outs['sequences'], skip_special_tokens=True)

            name = ['Holder', 'Target', 'Expression', 'Polarity']

            text = batch['text'][0]
            # print(text)
            a = scorer_tokenizer(text, return_tensors='pt')
            a = scorer_model(**a)
            tmp_scores = []
            for b in curr_outs_text:
                # print(b)
                b = scorer_tokenizer(b, return_tensors='pt')
                b = scorer_model(**b)
                score = (torch.matmul(a, b.t()).squeeze(dim=0).item()/768)
                tmp_scores.append(score)
            # print('-'*20)
            # print()
            tmp_scores = np.array(tmp_scores)
            tmp_scores = softmax(tmp_scores)
            tmp_scores = tmp_scores.tolist()
            index = tmp_scores.index(max(tmp_scores))

            if tmp_scores[index] - tmp_scores[0] < 0.7:
                index = 0

            # index = args.index
            dec = [curr_outs_text[index]]           # 0~7

            tmp_dec = dec[0]
            if ' [SSEP] ' in tmp_dec:
                tmp_pre = tmp_dec.split(' [SSEP] ')
            else:
                tmp_pre = [tmp_dec]

            tmp_pre_set = []        # 前缀的下一个 ['Target refers to A.', 'Target refers to B.']

            for sent in tmp_pre:
                if i == 0:
                    sub_sent = sent.split(f' {name[i+1]} refers to ')[0]
                elif i == 3:
                    sub_sent = sent.split(f'{name[i]} refers to')[1]
                    sub_sent = f'{name[i]} refers to' + sub_sent
                else:
                    sub_sent = sent.split(f' {name[i+1]} refers to ')[0]
                    sub_sent = sub_sent.split(f'{name[i]} refers to')[1]
                    sub_sent = f'{name[i]} refers to' + sub_sent
            tmp_pre_set.append(sub_sent)

            # 构造前缀
            # 之前的前缀分别加入当前的前缀，取最小值
            if len(pre_set):
                tps = []
                for i in range(min(len(pre_set), len(tmp_pre_set))):
                    tps.append(pre_set[i] + ' ' + tmp_pre_set[i])
                pre_set = tps.copy()
            else:
                pre_set = tmp_pre_set.copy()
            
            pre = batch['text'][0] + ' ' + ' [SSEP] '.join(pre_set)
            pre = tokenizer(pre, return_tensors="pt").input_ids

            all_outputs.append(curr_outs_text)
            target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        texts.append(batch['text'])
        outputs.extend(dec)
        targets.extend(target)

    print("\nPrint some results to check the sanity of generation method:", '\n', '-'*30)
    for i in [1, 5]: #, 25, 42, 50]:
        try:
            print(f'>>Target    : {targets[i]}')
            print(f'>>Generation: {outputs[i]}')
        except UnicodeEncodeError:
            print('Unable to print due to the coding error')
    print()

    scores, all_labels, all_preds = compute_scores(outputs, targets, dataset, seed, index)
    results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}

    return scores


# initialization
args = init_args()

# set random seed
seed_everything(args.seed)

print("\n", "=" * 30, f"NEW EXP: ASQP on {args.dataset}", "=" * 30, "\n")

# sanity check
# show one sample to check the code and the expected output
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# ----------------- #
print("current vocab size: ", len(tokenizer))
# ----------------- #
# tokenizer.add_tokens(['<prefix>'])
# tokenizer.add_tokens(['<suffix>'])
tokenizer.add_special_tokens({'additional_special_tokens': ['<prefix>', '<suffix>']})
# ----------------- #
print("current vocab size: ", len(tokenizer))
dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                      data_type='train', max_len=args.max_seq_length)
data_sample = dataset[7]  # a random data sample
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))

# training process
if args.do_train:
    print("\n****** Conduct Training ******")
    train_start = time.time()
    # initialize the T5 model
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    # tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir+"/darmstadt_unis/lightning_logs/version_24248/checkpoints/epoch=19-step=220.ckpt")
    # 加载训练好的参数
    # tfm_model.load_state_dict(torch.load("/home/zcl/work/dajie/ABSA-QUAD/outputs/darmstadt_unis/lightning_logs/version_24248/checkpoints/epoch=19-step=220.ckpt"))
    # 在加入新的token之后，需要重新初始化模型参数

    # 打印模型的词表大小
    print("tfm_model.config.vocab_size: ", tfm_model.config.vocab_size)

    tfm_model.resize_token_embeddings(len(tokenizer))
    print("tfm_model.config.vocab_size: ", tfm_model.config.vocab_size)
    model = T5FineTuner(args, tfm_model, tokenizer)

    # prepare for trainer
    if int(args.n_gpu) >= 1 or args.n_gpu == '-1':
        train_params = dict(
            accelerator='gpu',
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=2 if args.n_gpu == '-1' else int(args.n_gpu),
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=[LoggingCallback()],
            precision=16,
            auto_select_gpus=True,
            strategy='dp',
            fast_dev_run=args.fast_dev_run,
        )
    else:
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            callbacks=[LoggingCallback()],
            fast_dev_run=args.fast_dev_run,
        )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    print("Finish training and saving the model!")
    print(f"Training time: {time.time() - train_start:.2f} seconds")

    torch.save(model.state_dict(), f"./T5_Args/{args.dataset}/ssat5.pt")
else:
    print("\n****** Conduct Training ******")
    train_start = time.time()
    # initialize the T5 model
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    # tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir+"/darmstadt_unis/lightning_logs/version_24248/checkpoints/epoch=19-step=220.ckpt")
    # 加载训练好的参数
    # tfm_model.load_state_dict(torch.load("/home/zcl/work/dajie/ABSA-QUAD/outputs/darmstadt_unis/lightning_logs/version_24248/checkpoints/epoch=19-step=220.ckpt"))
    # 在加入新的token之后，需要重新初始化模型参数

    # 打印模型的词表大小
    print("tfm_model.config.vocab_size: ", tfm_model.config.vocab_size)

    tfm_model.resize_token_embeddings(len(tokenizer))
    print("tfm_model.config.vocab_size: ", tfm_model.config.vocab_size)
    model = T5FineTuner(args, tfm_model, tokenizer)
    model.load_state_dict(torch.load(f"./T5_Args/{args.dataset}/ssat5.pt"))

# evaluation
if args.do_direct_eval:
    print("\n****** Conduct Evaluating with the last state ******")
    eval_start = time.time()
    if args.ori:
        sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt', silence=True)
    else:
        sents, _, texts = read_line_examples_from_file(f'../data/{args.dataset}/test.json', True)

    dev_dataset = ABSADataset(tokenizer, data_dir=args.dataset,
                              data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, num_workers=4)

    # compute the performance scores
    scores = evaluate(test_loader, model, args)

    # write to file
    log_file_path = f"results_log/{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Datset={args.dataset}; Train bs={args.train_batch_size}, num_epochs = {args.num_train_epochs}," \
                   f"lr={args.learning_rate},eval_bs={args.eval_batch_size}," \
                   f"gradient_accumulation_steps={args.gradient_accumulation_steps},"
    exp_results = f"F1 = {scores['f1']:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)

    print("Finish evaluating!")
    print(f"Evaluating time: {time.time() - eval_start:.2f} seconds")
    print(f"Total time: {time.time() - train_start:.2f} seconds")