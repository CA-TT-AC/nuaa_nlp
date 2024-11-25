# NUAA_NLP experiment 2

学号 162110230   
姓名 徐敬

在任务2中，利用：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset 数据集做文本分类任务。主要测试了pretrained model的基于prompt的zero shot能力，以及在该数据集training set上进行sft后的能力。

## Checkpoints

Some checkpoints can be found in [huggingface](https://huggingface.co/CATTAC/nuaa_nlp/tree/main).

## Dataset process
利用``toutiao-text-classfication-dataset/split.py``对数据集进行training和validation set的划分，规定前70%为训练集，后30%为验证集。

## Inference script
修改模型路径，准备好切分后的数据集并运行：
```
python infer_classification.py
```
该脚本可以可视化地看到模型的输出。

## Supervised finetuning

准备好预训练过的模型，准备好切分后的数据集，运行
```
bash step_sft_classificaition.sh
```

## Validation
准备好预训练过的模型，准备好切分后的数据集，运行
```
python validation_classification.py
```

# NUAA_NLP experiment 1

## Tokenizer
利用char-based的方法进行tokenizer的训练。修改``train_tokenizer.py``中的数据集路径，运行：
```
bash step_01.sh
```
## Pretrain
准备pretrain数据集，具体dataloader在``data.py``中修改。

根据需要调整step_02.sh中的参数和数据集路径，运行：
```
bash step_02.sh
```

## Finetune
准备finetune数据集，具体dataloader在``data.py``中修改。

根据需要调整step_sft.sh中的参数和数据集路径，运行：
```
bash step_sft.sh
```

## Inference
在``inference.py``中修改对应模型路径以及inference的文本和解码算法，运行：
```
python inference.py
```

## ceval
在``run_ceval``中修改模型路径以及相关设置，运行：
```
bash run_ceval.sh
```