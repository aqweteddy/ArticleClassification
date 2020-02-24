from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from util import DataProcessor, convert_examples_to_features, InputFeatures
from torch.utils.data import DataLoader
from typing import List
# robert-wwm-ext-base

class InferClassifier:
    def __init__(self, model_dir, checkpoint, device='cpu', labels=None, maxlen=256):
        self.labels = labels if labels else ['政治時事', 'ACG', '交通工具', '3C', '人際關係＆感情', '閒聊', '運動健身', '購物', '西斯',
       '影劇', '美妝', '其他', '食物', '音樂', '旅遊', '遊戲']
        self.model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=len(self.labels)).to(device)
        self.model.load_state_dict(torch.load(checkpoint))
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.maxlen = maxlen
        self.device = device

    def infer_one(self, text: str, topk: int=1):
        example = DataProcessor().get_test_examples([text])
        dataset = convert_examples_to_features(example, max_length=self.maxlen, tokenizer=self.tokenizer)
        
        batch = dataset[0]
        inp_id, seg_id, inp_mask = batch
        inp_id = inp_id.unsqueeze(0).to(self.device)
        seg_id = seg_id.unsqueeze(0).to(self.device)
        inp_mask = inp_mask.unsqueeze(0).to(self.device)
        logits = self.__infer_model(inp_id, seg_id, inp_mask)
        return [self.labels[k] for k in logits[0].argsort()[::-1][:topk]]

    def infer_batch(self, text: List[str], topk: int=1, batch_size=16):
        examples = DataProcessor().get_test_examples(text)
        dataset = convert_examples_to_features(examples, max_length=self.maxlen, tokenizer=self.tokenizer)
        loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        result = []

        for batch in loader:
            inp_id, seg_id, inp_mask = batch
            inp_id = inp_id.to(self.device)
            seg_id = seg_id.to(self.device)
            inp_mask = inp_mask.to(self.device)
            logits = self.__infer_model(inp_id, seg_id, inp_mask)
            tmp = (-logits).argsort(axis=1)
            tmp = tmp[:, :topk].tolist()
            result.extend(tmp)
        
        return result


    
    def __infer_model(self, inp_id, seg_id, inp_mask):
        with torch.no_grad():
            self.model.eval()
            logits = self.model(inp_id, seg_id, inp_mask)[0]
            logits = logits.cpu().detach().numpy()
        return logits

if __name__ == "__main__":
    ic = InferClassifier('../roberta_wwm/roberta_wwm_ext')
    # print(ic.infer_one('sdsdsds ddd'))
    print(ic.infer_batch(['ss dsdsd', '何 樂'], topk=3))