# -*- coding: utf-8 -*-

#MAcBERT模型+LoRA微调

class MacBERTLoRAClassifier:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None


    def foward(self, input_ids, attention_mask):
        pass
