import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import TrainingArguments

from data_utils import ABSADataset
from models import RankEncoder
from utils import Trainer,data_collator


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='ds', type=str,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--pretrain_model_path", default='/root/data2/bert-base-uncased', type=str,
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
    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=1024,
                        help="random seed for initialization")
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3,
                        help="hidden dropout probability")
    parser.add_argument('--hidden_size', type=int, default=768,
                        help="hidden size")

    # training details
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)

    # extra settings
    parser.add_argument("--ori", action='store_true', help="Whether to use the original data")

    # 额外增加的rankgen的参数
    parser.add_argument('--retriever_model_path', default='/root/data2/t5-base', type=str)
    parser.add_argument('--cache_dir', default="./rankgen_cache_dir", type=str)
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--beam_size', default=6, type=int)
    parser.add_argument('--num_tokens', default=20, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--mask', type=str)

    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"./outputs/{args.dataset}/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_dataset(tokenizer, type_path, args):
    return ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                       data_type=type_path, max_len=args.max_seq_length)


def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


def main():
    # initialization
    args = init_args()

    # set random seed
    seed_everything(args.seed)

    print("\n", "=" * 30, f"NEW EXP: ASQP on {args.dataset}", "=" * 30, "\n")

    # sanity check
    # show one sample to check the code and the expected output
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    # ----------------- #
    print("current vocab size: ", len(tokenizer))
    # ----------------- #
    tokenizer.add_special_tokens({'additional_special_tokens': ['<prefix>', '<suffix>']})
    # ----------------- #
    print("current vocab size: ", len(tokenizer))
    train_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                                data_type='train', max_len=args.max_seq_length)
    # data_sample = train_dataset[7]  # a random data sample
    # print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    # print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
    # 加载验证集和测试集
    dev_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                              data_type='dev', max_len=args.max_seq_length)
    test_dataset = ABSADataset(tokenizer=tokenizer, data_dir=args.dataset,
                               data_type='test', max_len=args.max_seq_length)

    # 定义训练参数
    train_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,  # log & save weights each logging_steps
        save_steps=500,  # log & save weights each logging_steps
        save_total_limit=3,  # limit the total amount of checkpoints, delete the older checkpoints
        # evaluation_strategy='steps',  # evaluate each `logging_steps`
        # eval_steps=100,  # evaluate each `logging_steps`
        report_to='none',  # disable wandb logging
        fp16=True,  # enable mixed-precision training
        fp16_opt_level='O1',  # for fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
        do_train=True,  # training
        do_eval=False,  # disable evaluation
        do_predict=False,  # disable prediction
        dataloader_num_workers=4,  # number of workers for the DataLoader
        remove_unused_columns=False,  # remove unused columns in the training and evaluation datasets
        learning_rate = args.learning_rate #3e-4
    )

    # 定义模型
    model = RankEncoder(args)
    model.encoder.resize_token_embeddings(len(tokenizer))   # resize the token embeddings

    # 定义训练器,使用自定义的训练函数

    trainer = Trainer(
        args.dataset,
        args.mask,
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=train_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        # compute_metrics=compute_metrics,  # the callback that computes metrics of interest
        data_collator=data_collator,  # the callback that collates batches
    )

    # 开始训练
    trainer.train()
    # 保存模型
    # trainer.save_model(args.output_dir)
    torch.save(model.state_dict(), args.output_dir+'/model.pth')
    # torch.save(model, args.output_dir+'/model.pth')
    # 保存tokenizer
    # tokenizer.save_pretrained(args.output_dir)

    # 评估模型
    # if args.do_eval:
        # trainer.evaluate()

    # 预测
    # if args.do_inference:
        # print(trainer.predict(test_dataset))


if __name__ == '__main__':
    main()
