import torch.nn as nn
from transformers import AutoModelForMaskedLM


class RankEncoder(nn.Module):
    def __init__(self, config):
        super(RankEncoder, self).__init__()
        self.encoder = AutoModelForMaskedLM.from_pretrained(config.pretrain_model_path).bert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.hidden_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data=torch.nn.init.xavier_uniform(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        input_ids = input_ids.cuda()
        if attention_mask != None:
            attention_mask = attention_mask.cuda()
        if token_type_ids != None:
            token_type_ids = token_type_ids.cuda()
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # print(logits.shape)
        return logits