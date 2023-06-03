import torch
from transformers import Trainer

MAXX = -100

# 自定义trainer的损失函数
def rank_loss(logits, pos_example, neg_examples, dataset, mask, mean_ration=1.0):
    # if 'mpqa' in dataset  or dataset == 'norec':
        # zpi = torch.sum(torch.exp(torch.matmul(logits, pos_example.t()).squeeze(dim=0)), dim=-1)
        # neg_pi_1 = torch.sum(
        #     torch.exp(torch.matmul(logits, neg_examples[0].t()).squeeze(dim=0)), dim=-1)
        # neg_pi_2 = torch.sum(
        #     torch.exp(torch.matmul(logits, neg_examples[1].t()).squeeze(dim=0)), dim=-1)
        # neg_pi_3 = torch.sum(
        #     torch.exp(torch.matmul(logits, neg_examples[2].t()).squeeze(dim=0)), dim=-1)
        # if mask == 'holder_target':
        #     zpi = zpi + neg_pi_2 + neg_pi_3
        # elif mask == 'expression':
        #     zpi = zpi + neg_pi_1 + neg_pi_3
        # elif mask == 'polarity':
        #     zpi = zpi + neg_pi_1 + neg_pi_2
        # else:
        #     zpi = zpi + neg_pi_1 + neg_pi_2 + neg_pi_3
        # p_ci = torch.cat(
        #     [torch.exp(torch.matmul(logits[i].view(1, -1), pos_example[i].view(-1, 1)).squeeze(dim=0)) for i in
        #     range(logits.shape[0])], dim=0)
    if 'ds' in dataset or dataset == 'mpqa' :
        zpi = torch.sum(torch.exp(torch.matmul(logits, pos_example.t()).squeeze(dim=0)/768), dim=-1)
        neg_pi_1 = torch.sum(
            torch.exp(torch.matmul(logits, neg_examples[0].t()).squeeze(dim=0)/768), dim=-1)
        neg_pi_2 = torch.sum(
            torch.exp(torch.matmul(logits, neg_examples[1].t()).squeeze(dim=0)/768), dim=-1)
        neg_pi_3 = torch.sum(
            torch.exp(torch.matmul(logits, neg_examples[2].t()).squeeze(dim=0)/768), dim=-1)
        if mask == 'holder_target':
            zpi = zpi + neg_pi_2 + neg_pi_3
        elif mask == 'expression':
            zpi = zpi + neg_pi_1 + neg_pi_3
        elif mask == 'polarity':
            zpi = zpi + neg_pi_1 + neg_pi_2
        else:
            zpi = zpi + neg_pi_1 + neg_pi_2 + neg_pi_3
        p_ci = torch.cat(
            [torch.exp(torch.matmul(logits[i].view(1, -1), pos_example[i].view(-1, 1)).squeeze(dim=0)/768) for i in
            range(logits.shape[0])], dim=0)

    # neg_pi = torch.sum(
    #     torch.tensor(
    #         [torch.exp(torch.matmul(logits, neg_examples[i].t()).squeeze(dim=0)) for i in range(len(neg_examples))]),
    #     dim=-1)
    # zpi = zpi + neg_pi
    

    p_ci_pi = p_ci / zpi
    # print(torch.log(p_ci_pi))
    # print(int(logits.shape[0] * mean_ration))
    a = torch.log(p_ci_pi)
    # a = torch.where(torch.isinf(a), torch.full_like(a, -50), a)
    # res = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
    # if len(torch.where(torch.isnan(a))[0]):
    #     x = torch.where(torch.isnan(a))[0][0]
    #     print(p_ci[x])
    #     print(zpi[x])
    #     print('-------------'*3)
    # if len(torch.where(torch.isinf(a))[0]):
    #     x = torch.where(torch.isinf(a))[0][0]
    #     print(p_ci[x])
    #     print(zpi[x])
    #     print('-------------'*3)
    # print(res)
    loss = -torch.sum(a) / int(logits.shape[0] * mean_ration)
    # if abs(loss) <= 1e-8:
    # print(p_ci)
    # print(zpi)
    # print('-'*50)
    # print()
    return loss


# 自定义trainer的训练函数
def rank_train(model, batch, device, dataset, mask, mean_ration=1.0):
    model.train()
    # print(batch)
    # source_ids, source_mask, target_ids, target_mask = batch["input_ids"], batch["attention_mask"], batch["target_ids"], \
    #     batch["target_mask"]
    # 构建输入，一个正例，三个负例
    source_ids = batch["input_ids"].to(device)
    source_mask = batch["attention_mask"].to(device)
    pos_example = batch["target_ids"].to(device)
    pos_mask = batch["target_mask"].to(device)
    neg_examples = batch["neg_examples"]
    neg_masks = batch["neg_masks"]

    prefix_logits = model(source_ids, source_mask)
    pos_logits = model(pos_example, pos_mask)
    neg_logits = [model(neg_examples[i], neg_masks[i]) for i in range(len(neg_examples))]
    loss = rank_loss(prefix_logits, pos_logits, neg_logits, dataset, mask, mean_ration=mean_ration)
    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    return loss


def data_collator(batch):
    # print(batch)
    source_ids = torch.stack([example['input_ids'] for example in batch])
    source_mask = torch.stack([example['attention_mask'] for example in batch])
    pos_example = torch.stack([example['target_ids'][0] for example in batch])
    pos_mask = torch.stack([example['target_mask'][0] for example in batch])
    neg_examples = [torch.stack([example['target_ids'][i] for example in batch]) for i in
                    range(1, len(batch[0]['target_ids']))]
    neg_masks = [torch.stack([example['target_mask'][i] for example in batch]) for i in
                 range(1, len(batch[0]['target_mask']))]
    # target_ids = [example['target_ids'] for example in batch]
    # target_mask = [example['target_mask'] for example in batch]
    return {"input_ids": source_ids, "attention_mask": source_mask, "target_ids": pos_example, "target_mask": pos_mask,
            "neg_examples": neg_examples, "neg_masks": neg_masks}


class Trainer(Trainer):
    # def training_step(self, model, inputs):
    #     print(inputs)
    #     loss = rank_train(model, inputs, self.optimizer, self.args.device, mean_ration=1.0)
    #     return loss.detach()
    def __init__(self, dataset, mask, **key):
        super(Trainer, self).__init__(**key)
        self.dataset = dataset
        self.mask = mask


    def compute_loss(self, model, inputs, return_outputs=False):
        loss = rank_train(model, inputs, self.args.device, self.dataset, self.mask, mean_ration=1.0)
        return (loss, inputs) if return_outputs else loss 