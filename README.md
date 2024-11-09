# NUAA_NLP experiment 1

学号 162110230   
姓名 徐敬

## Pretrain
准备pretrain数据集，具体dataloader在``data.py``中修改。

根据需要调整step_01.sh中的参数和数据集路径，运行：
```
bash step_01.sh
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