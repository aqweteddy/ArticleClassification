import torch
from typing import List
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset


class InputExample:
    def __init__(self, example_id, content, label=None):
        self.content = content
        self.label = label
        self.example_id = example_id
    
    def __repr__(self):
        return      f"""id: {self.example_id}
                        label: {self.label}
                        content: {self.content}"""

class InputFeatures(Dataset):
    def __init__(self):
        self.example_ids = []
        self.input_ids = []
        self.input_masks = []
        self.seg_ids = []
        self.labels = []
    
    def add(self, example_id, input_id, input_mask, seg_id, label=None):
        self.example_ids.append(example_id)
        self.input_ids.append(input_id)
        self.seg_ids.append(seg_id)
        self.input_masks.append(input_mask)
        if label is not None:
            self.labels.append(label)
    
    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, k):
        if self.labels:
            return (torch.tensor(self.input_ids[k]),
                    torch.tensor(self.seg_ids[k]),
                    torch.tensor(self.input_masks[k]),
                    torch.tensor(self.labels[k]))
        else:
            return (torch.tensor(self.input_ids[k]),
                    torch.tensor(self.seg_ids[k]),
                    torch.tensor(self.input_masks[k]))

class DataProcessor(object):
    def __init__(self, labels=None):
        self.labels = labels if labels else ['政治時事', 'ACG', '交通工具', '3C', '人際關係＆感情', '閒聊', '運動健身', '購物', '西斯', '影劇', '美妝', '其他', '食物', '音樂', '旅遊', '遊戲']

    def get_train_examples(self, contents, labels):
        return self.__create_examples(contents, labels)
    
    def get_test_examples(self, contents, labels=None):
        return self.__create_examples(contents, labels)
    
    def get_label_id(self, label):
        for idx, tmp in enumerate(self.labels):
            if tmp == label:
                return idx

    def __create_examples(self, contents, labels=None) -> List[InputExample]:
        examples = []
        if labels is None:
            labels = [None] * len(contents)
        for cnt, content, label in zip(range(len(contents)), contents, labels):
            if content.strip():
                examples.append(InputExample(
                    example_id=cnt,
                    content=content,
                    label=self.get_label_id(label) if label else None
                ))
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_id: int = 0)->List[InputFeatures]:

    features = InputFeatures()
    for example in examples:
        inp = tokenizer.encode_plus(text=example.content, add_special_tokens=True, max_length=max_length)
        inp_ids, type_ids = inp['input_ids'], inp['token_type_ids']
        attention_mask = inp['attention_mask']
        
        padding_length = max_length - len(inp_ids)
        inp_ids = inp_ids + ([pad_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        type_ids = type_ids + ([pad_id] * padding_length)

        assert len(inp_ids) == max_length
        assert len(type_ids) == max_length
        assert len(attention_mask) == max_length

        features.add(example_id=example.example_id,
                    label=example.label,
                    input_id=inp_ids, 
                    input_mask=attention_mask, 
                    seg_id=type_ids)
    return features
