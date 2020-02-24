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