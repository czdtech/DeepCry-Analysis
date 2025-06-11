# DeepCry-Analysis: 婴儿哭声识别系统

一个基于深度学习和迁移学习的先进神经网络系统，用于高精度地识别婴儿哭声背后的原因。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.5%25-brightgreen.svg)](https://github.com/skytells-research/DeepCry-Analysis)
[![F1--Score](https://img.shields.io/badge/Macro%20F1--Score-90.6%25-brightgreen.svg)](https://github.com/skytells-research/DeepCry-Analysis)

## 项目简介

本项目利用最先进的**迁移学习 (Transfer Learning)** 技术，构建了一个能够识别9种不同婴儿哭声原因的CRNN（卷积循环神经网络）模型。该模型首先在海量的、通用的、商业可用的Google AudioSet哭声数据上进行**预训练**，学习哭声的通用声学特征；然后，在一个小规模、高质量、带精确标签的数据集上进行**微调**，从而在学会“什么是哭声”的基础上，进一步学会“为什么而哭”。

这种两阶段的训练策略，使得模型即使在某些类别的样本非常稀少的情况下，依然能达到极高的识别精度。

## 最终模型性能

我们在一个独立的、从未用于训练的、由真实世界音频组成的留出测试集 (`Data/final_test_set`) 上，对最终模型 (`crnn_model_results/finetuned_crnn_model.pth`) 进行了评估，取得了SOTA（State-of-the-Art）级别的性能。

**最终测试集分类报告:**
```
              precision    recall  f1-score   support

  belly_pain     0.7500    1.0000    0.8571         3
     burping     1.0000    1.0000    1.0000         4
    cold_hot     0.3333    1.0000    0.5000         1
  discomfort     1.0000    0.6667    0.8000         6
      hungry     1.0000    0.9870    0.9935        77
      lonely     1.0000    1.0000    1.0000         2
      scared     1.0000    1.0000    1.0000         4
       tired     1.0000    1.0000    1.0000         6
     unknown     1.0000    1.0000    1.0000        17

    accuracy                         0.9750       120
   macro avg     0.8981    0.9615    0.9056       120
weighted avg     0.9882    0.9750    0.9781       120
```

## 如何复现我们的结果

请严格按照以下步骤，在WSL (Windows Subsystem for Linux) 环境中复现我们的完整训练和评估流程。

### 步骤 0: 环境准备
- 确保您已安装 Python 3.10+ 和 Git。
- 克隆本仓库到您的本地。

### 步骤 1: 安装依赖
在WSL终端中，运行以下命令来安装所有必需的Python库。
```bash
pip install -r requirements.txt
```
我们还需要一个系统工具`ffmpeg`用于音频处理。
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 步骤 2: 准备数据
本项目的数据分为两部分：用于预训练的海量弱标签数据，和用于微调的高质量精标数据。

**A. 准备微调数据 (本地)**
1.  将您自己的、已按类别分好文件夹的哭声数据，放入 `Data/` 目录下。
2.  从[Mendeley Data](https://data.mendeley.com/datasets/hbppd883sd/1)下载数据集，解压后放入 `external_data/mendeley_dataset/`。
3.  运行以下脚本，它会自动整合以上数据源，并进行少量增强，生成最终用于微调的数据集。
    ```bash
    python prepare_final_dataset.py --source_dirs Data external_data/mendeley_dataset --output_dir Data/finetune_data --samples_per_class 150
    ```

**B. 准备预训练数据 (从YouTube下载)**
1.  **准备下载工具**: 从[yt-dlp的GitHub Releases页面](https://github.com/yt-dlp/yt-dlp/releases/latest)下载 `yt-dlp.exe`，并将其放置在项目根目录。
2.  **生成下载指令**: 在WSL中运行以下脚本。它会解析AudioSet的元数据，并生成一个Windows批处理文件 `download_commands.bat`。
    ```bash
    python generate_download_commands.py
    ```
3.  **执行下载**: 打开一个**Windows命令提示符(cmd)**，进入项目目录，然后运行批处理文件。**这个过程会非常漫长。**
    ```cmd
    e:
    cd e:\projects\babydontcry\DeepCry-Analysis
    download_commands.bat
    ```

### 步骤 3: 预训练模型
当海量数据下载完成后，在WSL中运行以下脚本，对模型进行预训练。
```bash
python pretrain_crnn.py
```
这将生成一个名为 `crnn_pretrained_encoder.pth` 的预训练权重文件。

### 步骤 4: 微调模型
现在，我们加载预训练好的权重，并在我们自己的高质量数据上进行微调。
```bash
python finetune_crnn.py
```
这将生成我们最终的模型 `crnn_model_results/finetuned_crnn_model.pth`。

### 步骤 5: 评估最终模型
使用以下命令，在独立的真实世界测试集上评估我们最终模型的性能。
```bash
python evaluate_crnn.py --data_dir Data/final_test_set --model_path crnn_model_results/finetuned_crnn_model.pth
```

## 核心脚本说明
- `prepare_final_dataset.py`: 整合多个数据源，并通过数据增强，生成用于训练的均衡数据集。
- `generate_download_commands.py`: 解析AudioSet元数据，生成一个包含数千条下载命令的Windows批处理文件。
- `pretrain_crnn.py`: 实现CRNN自动编码器，在海量弱标签数据上进行自监督预训练。
- `train_crnn.py`: 包含CRNN模型的核心定义，以及一个标准的从零开始的训练流程。
- `finetune_crnn.py`: 加载预训练好的模型权重，并在小规模、高质量的精标数据上进行微调。
- `evaluate_crnn.py`: 在一个标准的、按文件夹分类的数据集上评估CRNN模型。
- `evaluate_crnn_on_jams.py`: 在一个使用JAMS文件进行标注的数据集上评估CRNN模型。

## 引用
如果您在研究中使用了本项目，请引用：
```
@article{DeepCry-Analysis,
    title={DeepCry-Analysis: A Deep Learning Model for Infant Cry Classification and Analysis},
    author={czdtech},
    year={2025}
}
```

## 许可
本项目采用 **Apache License 2.0** 许可。
