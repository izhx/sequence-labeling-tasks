# 几个序列标注任务

## 任务
1. 任务1：Supersense-Tagged. https://github.com/nert-nlp/streusle
2. 任务2：SRL 英语德语汉语. https://github.com/System-T/UniversalPropositions
3. 任务3：预测Sem tag, The Parallel Meaning Bank. https://pmb.let.rug.nl/

## 使用
写好配置文件，使用如下（默认加载`./dev/config/CONFIG_NAME.yml`的配置文件）：
```
python main.py -c=GPU_ID -y=CONFIG_NAME
```
或后台运行：
```
nohup python -u main.py -c=GPU_ID -y=CONFIG_NAME > ./dev/log/LOG_NAME &
```

## 结构
```
.
├── data  # depsawr 用的vocab类，因为pickle的原因，必须放在顶层
├── dev  # 存储实验文件的目录，如设置、日志、模型、数据cache、词向量等
├── nmnlp  # submodule, 代码脚手架，用于提供trainer等
├── notag  # 简化后的 depsawr 代码
├── pmb-3.0.0  # The Parallel Meaning Bank 数据
├── streusle  # submodule, Supersense 数据
├── UniversalPropositions  # submodule, 多语言SRL数据
├── datasets.py  # 数据读取相关代码
├── main.py  # 主要实验脚本，训练、测试等
├── metric.py  # 评价分数类
├── models.py  # 模型代码，srl、crf tagger和自己写的depsawr，有tag的，以后改改可以放脚手架里
├── README.md
├── translate.py  # GitHub上找的繁体转简体的脚本
└── util.py  # 工具代码，根据存的测试结果算F1的sql工具

```