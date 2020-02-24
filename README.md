# article classification

## categories

* ['政治時事', 'ACG', '交通工具', '3C', '人際關係＆感情', '閒聊', '運動健身', '購物', '西斯',
       '影劇', '美妝', '其他', '食物', '音樂', '旅遊', '遊戲']

## Model

## Roberta-wwm-base

* 3 epoch
* 8 batch-size
* 256 seq len
* merge_train.csv

```sh
test_acc: 0.8986830356015696    test_loss: 0.3279069839544685
train_acc: 0.8542825410771189   train_loss: 1.083221197128296

test_acc: 0.9050314390484876    test_loss: 0.3094365964986496
train_acc: 0.9224940363300494   train_loss: 1.0304343700408936

test_acc: 0.9013105876177184 test_loss: 0.3560551377999998
train_acc: 0.9610454063984175   train_loss: 0.013738512992858887
```

## Distilbert

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

## Server

* flask
* `[HOST]/infer_one_class?text=[TEXT]&topk=[TOPK]`

### Performance

#### On CPU AMD R5-3600

* 100items / min

#### On GPU RTX2070

* 38.62 items / sec