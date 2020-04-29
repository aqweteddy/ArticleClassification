import os
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.utils import shuffle
from sklearn.metrics import precision_score
import numpy as np
from util import DataProcessor, convert_examples_to_features, InputFeatures


def evaluate(loader, model_dir, ckpt, num_labels):
    loss = 0.0
    nb_eval_steps = 0
    y_pred = None
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(model_dir, ckpt)))
    model = model.cuda()

    for batch in tqdm(loader, desc="Evaluating"):
        model.eval()
        inp_ids, seg_ids, inp_masks, labels = batch
        inp_ids = inp_ids.cuda()
        seg_ids = seg_ids.cuda()
        inp_masks = inp_masks.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            tmp_loss, logits = model(inp_ids, seg_ids, inp_masks, labels=labels)
            loss += tmp_loss.mean().item()
            nb_eval_steps += 1

            if y_pred is None:
                y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
                y_true = labels.detach().cpu().numpy()
            else:
                y_pred = np.append(y_pred, np.argmax(logits.detach().cpu().numpy(), axis=1))
                y_true = np.append(y_true, labels.detach().cpu().numpy())
    
    loss = loss / nb_eval_steps
    acc = precision_score(y_true, y_pred, average='weighted')
    print(f"test_acc: {acc}\ttest_loss: {loss}")
    return acc, loss


def read_data(file):
    df = pd.read_json(file)
    df = shuffle(df)
    content = (df['title'] + ' ' + df['content']).to_list()
    target = df['category'].to_list()
    return content, target

if __name__ == '__main__':
    import pandas as pd


    NUM_TEST_DATA = 50016
    MODEL_DIR = './roberta_wwm_ext'
    MAX_LEN = 512
    BATCH_SIZE = 16 * 2 # 8gpu * 16
    NUM_LABELS = 33

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    content, target = read_data('../../corpus/ettoday_2017.json')
    examples = DataProcessor().get_test_examples(content[:NUM_TEST_DATA], target[:NUM_TEST_DATA])
    test_dataset = convert_examples_to_features(examples, max_length=MAX_LEN, tokenizer=BertTokenizerFast.from_pretrained(MODEL_DIR))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    evaluate(test_loader, MODEL_DIR, 'step_28124.ckpt', NUM_LABELS)