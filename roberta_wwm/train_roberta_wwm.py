from util import DataProcessor, convert_examples_to_features, InputFeatures
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import precision_score
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

writer = SummaryWriter('./tfboard_log')


def evaluate(loader, model):
    loss = 0.0
    nb_eval_steps = 0
    y_pred = None
    for batch in tqdm(loader, desc="Evaluating"):
        model.eval()
        inp_ids, seg_ids, inp_masks, labels = batch
        inp_ids = inp_ids.to(DEVICE)
        seg_ids = seg_ids.to(DEVICE)
        inp_masks = inp_masks.to(DEVICE)
        labels = labels.to(DEVICE)
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




def train(loader, model_dir, lr=2e-5, num_labels=18, epochs=4, save_steps=3000, test_loader=None, eval_callback=None):
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model = model.to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    global_step = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        ys_true = []
        ys_pred = []
        for i, batch in enumerate(tqdm(loader, desc='Training')):
            inp_ids, seg_ids, inp_masks, labels = batch
            inp_ids = inp_ids.to(DEVICE)
            seg_ids = seg_ids.to(DEVICE)
            inp_masks = inp_masks.to(DEVICE)
            labels = labels.to(DEVICE)

            # tensorboard
            if epoch == 0 and i == 0:
                writer.add_graph(model, (inp_ids, seg_ids))

            loss, logits = model(inp_ids, seg_ids, inp_masks, labels=labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            ys_true.extend(labels.to('cpu').detach().numpy())
            logits = logits.to('cpu').detach().numpy()
            ys_pred.extend(np.argmax(logits, axis=1))
            
            # tensorboard
            if global_step % 100 == 0:
                writer.add_scalar(f'Train/Loss-Epoch', loss.mean().item(), global_step)
                writer.add_scalar(f'Train/Accuracy', precision_score(ys_true, ys_pred, average='weighted'), global_step)
                writer.flush()

            if global_step % 5000 == 0:
                if eval_callback:
                    test_acc, test_loss = eval_callback(test_loader, model)
                train_acc = precision_score(ys_true, ys_pred, average='weighted')
                train_loss = loss.mean().item()
                writer.add_scalars('TrainTest/Loss', {'train': train_loss, 'test': test_loss}, global_step)
                writer.add_scalars('TrainTest/Accuracy', {'train': train_acc, 'test': test_acc}, global_step)
                writer.flush()
                print(f"step: {global_step}\ttrain_acc: {train_acc}\ttrain_loss: {train_loss}")
                torch.save(model.state_dict(), f'{model_dir}/step_{global_step}.ckpt')
            
            global_step += 1

    torch.save(model.state_dict(), f'{model_dir}/final.ckpt')



def read_data(file):
    df = pd.read_csv(file)
    df = shuffle(df)
    content = (df['raw_title'] + ' ' + df['raw_text']).to_list()
    target = df['category'].to_list()
    return content, target

def main():
    NUM_TRAIN_DATA = 85000
    MODEL_DIR = './roberta_wwm_ext'
    MAX_LEN = 256
    BATCH_SIZE = 8
    NUM_LABELS = 16
    EPOCHS = 3

    # read data
    content, target = read_data('../fetch_data/merge_train.csv')

    # train dataloader
    examples = DataProcessor().get_train_examples(content[:NUM_TRAIN_DATA], target[:NUM_TRAIN_DATA])
    train_dataset = convert_examples_to_features(examples, max_length=MAX_LEN, tokenizer=BertTokenizer.from_pretrained(MODEL_DIR))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # test dataloader
    examples = DataProcessor().get_test_examples(content[NUM_TRAIN_DATA:], target[NUM_TRAIN_DATA:])
    test_dataset = convert_examples_to_features(examples, max_length=MAX_LEN, tokenizer=BertTokenizer.from_pretrained(MODEL_DIR))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

    # start training and callback for eval
    # train(train_loader, MODEL_DIR, num_labels=NUM_LABELS, epochs=EPOCHS, eval_callback=evaluate, test_loader=train_loader)
    train(train_loader, MODEL_DIR, num_labels=NUM_LABELS, epochs=EPOCHS, eval_callback=evaluate, test_loader=test_loader)


if __name__ == '__main__':
    main()
    # writer.export_scalars_to_json("./log.json")
    writer.close()