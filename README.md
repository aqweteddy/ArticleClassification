# article classification

## categories

* ['政治時事', 'ACG', '交通工具', '3C', '人際關係＆感情', '閒聊', '運動健身', '購物', '西斯',
       '影劇', '美妝', '其他', '食物', '音樂', '旅遊', '遊戲']

## Data

* dcard
* ptt
* mobile01
* gamer

## Model

### electra

#### 3 epochs

* 3 epochs
* 8 * 8 batch_size
* 512 seq len
* merge_train.csv

```sh
train_acc: 0.6227425686576363, test_acc: 0.8314032920798782, train_loss: 0.4438794255256653, test_loss: 0.628982390325094
train_acc: 0.8720725626728351, test_acc: 0.8701059766006879, train_loss: 0.4975391626358032, test_loss: 0.43683170980399416
train_acc: 0.9063946957973625, test_acc: 0.8791534506241079, train_loss: 0.27478697896003723, test_loss: 0.40323783879734804
train_acc: 0.933983243834291, test_acc: 0.8828887036180814, train_loss: 0.13545885682106018, test_loss: 0.4126528928958878
train_acc: 0.9537589253084459, test_acc: 0.8857686740350325, train_loss: 0.19591882824897766, test_loss: 0.42418756520318
...
train_acc: 0.9841, test_acc: 0.8849, train_loss: 0.02976, test_loss: 0.4857
```

### Roberta-wwm-base

* (Pretrained model link)[https://github.com/ymcui/Chinese-BERT-wwm]
* 3 epoch
* 8 batch-size
* 256 seq len
* merge_train.csv

```sh
test_acc: 0.8986830356015696    test_loss: 0.3279069839544685
train_acc: 0.8542825410771189   train_loss: 1.083221197128296

test_acc: 0.9050314390484876    test_loss: 0.3094365964986496
train_acc: 0.9224940363300494   train_loss: 1.0304343700408936

test_acc: **0.906**3105876177184    test_loss: 0.3560551377999998
train_acc: 0.9610454063984175   train_loss: 0.013738512992858887
```

### Distilbert

* 4epoch
* 16 batch size
* 512 seq length

```sh
test_acc: 0.7993509267328244    test_loss: 0.6088341759096373
train_acc: 0.7381339957782092   train_loss: 0.541751503944397

test_acc: 0.8231009700852281    test_loss: 0.5566842174804307
train_acc: 0.8358543512966032   train_loss: 1.0290946960449219

test_acc: 0.8281494867214262    test_loss: 0.5552177424754902
train_acc: 0.9007723933699896   train_loss: 0.14890894293785095

test_acc: 0.821698865708592     test_loss: 0.6597533390897358
train_acc: 0.9482712985261998   train_loss: 0.010299921035766602
```

### Performance

#### On CPU AMD R5-3600

* 100items / min

#### On GPU RTX2070

* 38.62 items / sec
