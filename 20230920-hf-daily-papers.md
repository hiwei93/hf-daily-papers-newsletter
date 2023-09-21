# 【2023-09-20】Huggingface 每日论文速览

- [Huggingface Daily Papers 2023-09-20](https://huggingface.co/papers?date=2023-09-20) 共推荐 9 篇论文。

> 💡说明：
>
> - 本文对 Huggingface Daily Papers 推荐的论文从：主要工作、主要两点、关键词和评估四个方面进行速览。
>
> - 论文的速览内容基于论文的摘要，使用 GPT-4 进行内容生成，然后使用程序将内容整合，并以 Markdown 文本呈现。

## Language Modeling Is Compression

**1. 介绍本文的主要工作**

本研究聚焦于预测模型与无损压缩器的等价关系，详细研究了大型语言模型的压缩能力。作者通过将预测问题视为压缩问题，揭示了大型语言模型作为强大的通用预测器的能力。其阐释了从压缩视角解读如：缩放律、标记化和在上下文中学习等方面的新见解。文章也提出了Chinchilla 70B模型能出色地压缩ImageNet图像片段和LibriSpeech样本，其性能优于领域特定的压缩器。

**2. 本文工作的主要亮点**

首先，该研究通过将预测问题重新框定为压缩问题，倾向于从新的视角观察大型语言模型的性能。其次，作者发现，尽管训练主要基于文本，Chinchilla 70B模型却能将ImageNet图像片段和LibriSpeech样本压缩到原始大小的43.4%和16.4%，超越了专门的领域压缩器的性能。最后，这篇文章证明预测-压缩等价关系使得我们可以使用任何压缩器（例如gzip）来构建条件生成模型，这为模型拓展提供了可能。

**3. 核心关键词**

- `Predictive models` (预测模型)

- `Compression` (压缩)

- `Large language models` (大型语言模型)

- `Chinchilla 70B` (Chinchilla 70B)

- `Generative models` (生成模型)

**4. 从实用性、创新性和推荐度进行打分**

- 实用性：4分。该研究的成果可以涵盖多领域，不仅限于语言模型，也可以适用于图像和音频等领域，具有广泛应用性，但对于非科研工作者可能理解稍有难度。

- 创新性：4.5分。文章的观点颇具创新精神，有助于开辟新的研究方向，并为预测模型的压缩功能和条件生成模型提供了新的视角。

- 推荐度：4分。对于机器学习、自然语言处理和压缩算法的研究者，这篇论文给出的新视角和研究结果极具参考价值。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10668)

## Multimodal Foundation Models: From Specialists to General-Purpose  Assistants

### 论文总结

**1. 本文主要工作**

这篇文章对多模态基础模型的分类和演变进行了全面的调研，特别关注了从专业模型到通用助手的转变。包括对既定研究领域如对特定目标进行预训练的多模态基础模型进行评估，以及对探索性、尚在开放研究的领域进行概述。这篇文章的目标读者是计算机视觉和视觉-语言多模态社区的研究人员、研究生和专业人士。

**2. 本文的主要亮点**

这篇文章的亮点在于它旨在桥接多模态基础模型的专门化和通用化。它提供了一个关于此领域持续发展的全面且深思熟虑的观察，包括视觉理解、文本到图像的生成、大型语言模型（LLM）的统一视觉模型、LLM的端到端训练，以及链式多模态工具与LLM。

**3. 核心关键词**

- `Multimodal Foundation Models` (多模态基础模型)

- `Visual Understanding` (视觉理解)

- `Text-to-Image Generation` (文本到图像生成)

- `Large Language Models` (大型语言模型)

- `End-to-End Training` (端到端训练)

**4. 评分**

- 实用性：4/5

- 创新性：4/5

- 推荐度：4/5

这篇文章提供了多模态基础模型领域的全面视角，并对其演变进行了深入分析。尽管它在实用性和创新性方面都相当高，但有待进一步实验细节和实务应用的示例来更充分地证明其效果。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10020)

## OpenBA: An Open-sourced 15B Bilingual Asymmetric seq2seq Model  Pre-trained from Scratch

**论文总结**

**1. 本文主要工作**

本研究论文介绍了OpenBA，一种开源的15B双语不对称seq2seq模型，旨在贡献给以中文为主导的开源模型社区。作者采用了有效和高效的技术，并采用了三阶段的训练策略从零开始训练模型。研究结果显示，仅使用380B的tokens，其性能就超过了各项基准测试，如BELEBELE，MMLU，以及C-Eval (hard)。

**2. 本文工作的主要亮点**

1. 通过有效和高效的技术，采用三阶段训练策略，实现了从零开始训练模型。

2. 在380B tokens的训练下，OpenBA表现出色，打败了多项基准测试。

3. 结合了预训练数据处理、双语Flan数据收集、启发模型架构设计的经验观察、不同阶段的训练目标以及其他优化技术。

4. 代码已按照Huggingface Transformers库的设计原则进行重构，易于开发者使用并已发布不同阶段的训练检查点。

**3. 核心关键词**

- `OpenBA` (OpenBA)

- `Seq2seq model` (序列到序列模型)

- `Bilingual Flan data` (双语Flan数据)

- `Pre-training data processing` (预训练数据处理)

- `Three-stage training strategy` (三阶段训练策略)

**4. 评分**

- **实用性**：5/5。OpenBA模型能在多项基准测试上表现出色，并且代码已对开发者友好，实用性很强。

- **创新性**：4.5/5。本文提出了一种有效的三阶段训练策略，并从零开始训练模型，具有一定的创新性。

- **推荐度**：5/5。该模型在性能和实用性上表现出色，同时提供了详尽的实现细节，具有很高的

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10706)

## Baichuan 2: Open Large-scale Language Models

1. **介绍本文的主要工作**

   本文主要介绍了一系列名为 Baichuan 2 的大型多语言语言模型。这些模型包含了70亿和130亿的参数，通过全新的训练算法，使用2.6万亿的tokens进行训练。Baichuan 2 在公开的基准测试，如MMLU、CMMLU、GSM8K、和HumanEval中，或达到或超过了类似大小的其他开源模型的性能。特别地，Baichuan 2 在医学和法律等垂直领域表现出色。作者们将会发布所有预训练模型的检查点，以利于研究社区更好的理解 Baichuan 2 的训练动态。

2. **本文工作的主要亮点**

   - Baichuan 2 是一种大型多语言语言模型，具有大规模的参数和语料训练；

   - 在一系列公开的基准测试中，Baichuan 2 的性能超过了类似大小的其他开源模型；

   - Baichuan 2 在医学和法律等垂直领域表现优秀；

   - 作者们承诺将公开模型的训练检查点，为研究社区提供了有价值的资源。

3. **核心关键词**

   - Large Language Models (大型语言模型)

   - Multilingual (多语言)

   - Pre-training Model Checkpoints (预训练模型检查点)

   - Public Benchmarks (公开基准测试)

   - Vertical Domains (垂直领域)

4. **评分**

   - 实用性：5分

   - 创新性：4分

   - 推荐度：5分

   由于Baichuan 2 不仅在公开的基准测试上表现出色，而且在医学和法律等具有实际重要性的垂直领域中，也有出色的应用潜力，因此，在实用性上给出满分。考虑到Baichuan 2 所采用的是现有的大语言模型技术，虽然它有创新的点，如使用大规模参数和语料进行训练，但在创新性上给出4分。基于它在各个测试基准上的优秀表现，并且作者承诺将公开模型的预训练检查点，为研究社区提供了很大的帮助，因此在推荐度上给出满分。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10305)

## SlimPajama-DC: Understanding Data Combinations for LLM Training

### 介绍本文的主要工作

本文针对利用多源数据集SlimPajama训练大型语言模型的问题，进行了名为SlimPajama-DC的实证分析，以揭示使用SlimPajama进行训练的基本特性和最佳实践。研究过程中，作者对两项关键问题进行了深入探索，一是全局去重和局部去重对模型性能的影响；二是高质量、高度去重的多源数据集在结合中的占比。为了进行研究，构建了六种配置的SlimPajama数据集，并使用1.3B Cerebras-GPT模型进行训练，发现最佳配置在使用相同数量训练token的情况下，明显优于使用RedPajama训练的1.3B模型。

### 本文工作的主要亮点

本研究使用了从大规模1.2T tokens的RedPajama数据集精炼并进一步去重到627B tokens的复杂、已严格去重的SlimPajama多源数据集。此外，它通过构建多种数据集配置，以实证方式深入分析了全局去重和局部去重以及数据集多样性对模型性能的影响。并且，最优的配置显著地优于使用同样数量的训练tokens进行训练的RedPajama模型。

### 核心关键词

- SlimPajama (`SlimPajama`)

- Large Language Models (`大型语言模型`)

- Deduplication (`去重`)

- Multi-source Dataset (`多源数据集`)

- Language Model Training (`语言模型训练`)

### 评分

实用性：4.5分  

模型性能受到数据组合影响的研究为训练大型语言模型提供了新的实践视角。

创新性：4分  

提出了全局去重和局部去重的概念，并针对其对模型性能的影响进行了深入研究，设计了多种数据集配置来揭示使用SlimPajama训练的最佳实践。

推荐度：4分  

鉴于其在语言模型训练中的实用性和创新性，对任何对数据组合影响模型性能感兴趣的读者都值得一读。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10818)

## Q-Transformer: Scalable Offline Reinforcement Learning via  Autoregressive Q-Functions

**文章总结：**

1. **本文的主要工作**

   本文提出了一种可扩展的强化学习方法 Q-Transformer，用于从大型离线数据集中训练多任务策略。该方法既可以利用人类的示范，也可以利用自主收集的数据。通过使用 Transformer 来为 Q-functions 提供一种可扩展的表示，通过对每个动作维度进行离散化，并表示每个动作维度的 Q-值作为单独的令牌，能够将高容量的序列建模技术应用到 Q-learning 中。

2. **本文工作的主要亮点**

   

   Q-Transformer 利用 Transformer 和高容量序列建模技术，实现了 Q 函数的离线时间差分备份训练，使得强化学习更加高效和可扩展。此外，该方法在大规模多样化的真实世界机器人操纵任务套件上，超过了先前的离线 RL 算法和模仿学习技术。

3. **核心关键词**

    - Q-Transformer (`Q-Transformer`)

    - Reinforcement learning (`强化学习`)

    - Offline training (`离线训练`)

    - Q-functions (`Q 函数`)

    - Transformers (`变压器模型`)

4. **评分**

   - 实用性 : 4.5/5

   - 创新性 : 4/5

   - 推荐度 : 4.5/5

   本文的 Q-Transformer 显示了在处理多任务策略训练的有效性与灵活性，相当具有实用性。而且，将强化学习与 Transformer 结合，属于典型的交叉学科创新。由于其在多项实际任务中的出色表现，本文值得大家

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10150)

## Stabilizing RLHF through Advantage Model and Selective Rehearsal

### 论文总结

1. **主要工作**  

这篇技术报告解决了在使用强化学习从人类行为学习（RLHF）来对齐大的语言模型（LLMs）时遇到的稳定性问题。本文提出了两个创新方法：（1）优势模型 (Advantage Model)，该模型直接建模优势得分，即额外的报酬与预期报酬相比，并在任务之间调节得分分布以防止奖励黑客；（2）选择性复习 (Selective Rehearsal)，该方法通过策略性地选择数据进行PPO训练和知识复习来减轻灾难性的遗忘。

2. **主要亮点**  

本文的主要亮点在于其提出的两个解决对齐大的语言模型稳定性问题的创新方法。这些方法显著提高了强化学习从人类行为学习的稳定性，并实现了更高的奖励得分和胜率。这些技术的实用性和创新性都得到了实验证明。

3. **核心关键词**  

- `RLHF` (强化学习从人类行为学习)

- `LLMs` (大的语言模型)

- `Advantage Model` (优势模型)

- `Selective Rehearsal` (选择性复习)

- `PPO` (脉冲编程优化)

4. **评分**

   - 实用性：5/5

   - 创新性：5/5

   - 推荐度：5/5

这篇技术报告成功解决了强化学习从人类行为学习进行大语言模型调整中常见的稳定性问题，并展示了卓越的实用性、创新性和推荐度。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10202)

## 360^circ Reconstruction From a Single Image Using Space Carved  Outpainting

### 文章总结

**1. 本文的主要工作**

本文介绍了一个名为POP3D的新型框架，它可以从单一图像创建全方位360°视图的3D模型。不仅能够处理多种类型的图像，还提高了重建效果的准确性和自然性。

**2. 本文工作的主要亮点**

POP3D的主要亮点在于其显著的泛化能力和提供高质量的重构能力。其主要组件包括：独眼深度和法线预测器，空间刻画法用于描绘目标物体可能未被观察到的部分，预先在大型图像数据集上训练的生成模型，以及专门用于使用RGB图像和单目几何线索重构物体的神经隐式表面重构方法。

**3. 核心关键词**

- `POP3D` (`POP3D`)

- `Monocular Depth and Normal Predictor` (`单目深度和法线预测器`)

- `Space Carving` (`空间刻画`)

- `Generative Model` (`生成模型`)

- `Neural Implicit Surface Reconstruction` (`神经隐式表面重构`)

**4. 评分**

- 实用性：4.5/5

- 创新性：5/5

- 推荐度：5/5

POP3D的实用性很高，因为它实现了从单一图像创建360°视图的3D模型。在研究领域，它极具创新性，因为它成功地解决了之前方法所面临的难以应对多种类型和提供高质量重构的问题。我很推荐这篇文章，因为它在类似的工作中表现出了显著的优势，取得了最先进的重构效果。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10279)

## FoleyGen: Visually-Guided Audio Generation

**一、本文的主要工作**

本文介绍了一种名为FoleyGen的视觉引导音频生成系统。在处理视频至音频（V2A）生成任务的挑战中，FoleyGen采用了语言建模范式，使用了一个开放领域的音频编解码器用于波形和离散标记之间的双向转换。为解决在视觉动作与音频之间对齐的问题，系统同时还探索了三种新颖的视觉注意力机制。

**二、本文工作的主要亮点**

该系统的主要亮点在于：

1. 采用了开放领域的神经音频编解码器，可双向转换波形与离散标记，解决了视频至音频(V2A)生成的华波兰挑战。

2. 创新性地探索了三种新颖视觉注意力机制，解决了音频生成与视频视觉行为之间的时间不同步问题。

3. 采用不同的视觉编码器对系统进行了全面评估。这些编码器在单模态或多模态任务上均预先进行了训练。

4. 系统在VGGSound数据集上的实验结果超过了之前的系统，同时在所有客观指标和人类评估中都表现优异。

**三、核心关键词**

1. `FoleyGen` (FoleyGen系统)

2. `V2A Generation` (视频至音频生成)

3. `Neural Audio Codec` (神经音频编解码器)

4. `Visual Attention Mechanisms` (视觉注意力机制)

5. `Visual Encoder` (视觉编码器)

**四、实用性、创新性和推荐度打分**

1. **实用性打分**: 4.5/5 - 该论文解决的问题在现实生活中具有广泛的应用场景，例如在视频编辑、将音频机器转换为视频内容等方面都有广泛的应用。

2. **创新性打分**: 5/5 - 该论文探索的三种视觉注意力机制，以及利用开放领域的神经音频编解码器进行音频生成，都体现出了较高的创新性。

3. **推荐度打分**: 4.7/5 - 基于实验结果在所有客观指标和人类评估中表现出优异，以及该系统够处理一个难以解决的问题，我会强烈推荐这篇文章。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.10537)