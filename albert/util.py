import torch
from typing import List
from tqdm import tqdm
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
    
    def add(self, example_id, input_id, input_mask, seg_id, label):
        self.example_ids.append(example_id)
        self.input_ids.append(input_id)
        self.seg_ids.append(seg_id)
        self.input_masks.append(input_mask)
        self.labels.append(label)
    
    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, k):
        return (torch.tensor(self.input_ids[k]),
                torch.tensor(self.seg_ids[k]),
                torch.tensor(self.input_masks[k]),
                torch.tensor(self.labels[k])  if self.labels else None)
    

class DataProcessor(object):
    def get_train_examples(self, contents, labels):
        return self.__create_examples(contents, labels)
    
    def get_test_examples(self, contents, labels):
        return self.__create_examples(contents, labels)
    
    def get_label_id(self, label):
        labels = ['財經', '地方', '軍武', '社會', '3C家電', '親子', 'ET車雲', '影劇', '生活', '國際',
       '政治', '旅遊', '大陸', '體育', '健康', 'ET來了', '寵物動物', '新奇', '保險', '時尚',
       '房產雲', '論壇', '消費', '法律', '遊戲', '網搜', '名家', '公益', '男女', '電影', '電商',
       '公民(勿用)', '運勢']
        for idx, tmp in enumerate(labels):
            if tmp == label:
                return idx

    def __create_examples(self, contents, labels=None) -> List[InputExample]:
        examples = []
        if labels is None:
            labels = [None] * len(contents)
        for cnt, content, label in zip(range(len(contents)), contents, labels):
            if content.strip():
                try:
                    examples.append(InputExample(
                        example_id=cnt,
                        content=content,
                        label=self.get_label_id(label)
                    ))
                except Exception:
                    pass
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_id: int = 0)->List[InputFeatures]:

    features = InputFeatures()
    for example in tqdm(examples, desc="convert examples to features"):
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



if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.utils import shuffle
    from transformers import BertTokenizer

    
    df = pd.read_csv('../dcard.csv')
    df = shuffle(df)
    content = (df['raw_title'] + df['raw_text']).to_numpy()
    target = df['category'].to_numpy()

    examples = DataProcessor().get_test_examples(content[:10])
    result = convert_examples_to_features(examples, max_length=256, tokenizer=BertTokenizer.from_pretrained('./rbtl3'))
    
    print(result[0])