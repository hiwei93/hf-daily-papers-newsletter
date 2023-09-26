# 【2023-09-22】Huggingface 每日论文速览

- [Huggingface Daily Papers 2023-09-22](https://huggingface.co/papers?date=2023-09-22) 共推荐 8 篇论文。

> 💡说明：
>
> - 本文对 Huggingface Daily Papers 推荐的论文从：主要工作、主要两点、关键词和评估四个方面进行速览。
>
> - 论文的速览内容基于论文的摘要，使用 GPT-4 进行内容生成，然后使用程序将内容整合，并以 Markdown 文本呈现。

## LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models

### 1. 介绍本文的主要工作

本文提出了一种称为LongLoRA的短期稀疏注意力和LoRA的参数有效的微调方法，使得大型语言模型（LLMs）能够有效地扩展其上下文范围，但计算成本低。虽然密集的全局注意力在推理中是必需的，但模型的微调可以通过稀疏的局部注意力有效有效完成。

### 2. 本文工作的主要亮点

LongLoRA的主要亮点在于，在扩展上下文的同时，能显著地降低计算成本。特别是它所提出的轮换短期注意有效地实现了上下文的扩展，能够在训练中省去大量的计算，而且可以用仅两行代码实现。此外，本文通过一个类似LoRA的方法重新审视了上下文扩展的参数微调机制，发现这种微调方式在可训练的嵌入和标准化的前提下效果良好。

### 3. 核心关键词

- `LongLoRA` (`长LoRA`)

- `Large Language Models` (`大型语言模型`)

- `Sparse Local Attention` (`稀疏局部注意力`)

- `Context Extension` (`上下文扩展`)

- `Efficient Fine-tuning` (`高效微调`)

  

### 4. 评分

- 实用性：4.5/5。 LongLoRA能够有效地扩展语言模型的上下文范围，而且计算成本低，具有很高的实用价值。

- 创新性：4/5。 本文提出了一种新的微调方式，通过轮换短期注意力与参数有效的调整相结合，提高了模型在处理长上下文的能力，相当具有创新性。

- 推荐度：4.5/5。 对于需要使用大型语言模型处理长上下文任务的研究者或从业人员，本文值得一读。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.12307)

## RMT: Retentive Networks Meet Vision Transformers

**1. 本文的主要工作**

本文提出了一种新的计算机视觉模型RMT，它将Retentive Network与Transformer结合在一起。RMT引入了明显的衰减，使视觉模型能够获取关于空间距离的先验知识。此外，为了减少全局建模的计算成本，作者沿着图片的两个坐标轴将建模过程分解。广泛的实验表明RMT在多种计算机视觉任务上表现出色。

**2. 本文工作的主要亮点**

- RMT是首个将RetNet思想应用到计算机视觉中的尝试，并取得了非常好的效果。

- 作者探讨了如何减少全局建模的计算成本，提出了沿图像两个坐标轴进行建模分解的方法。

- RMT在众多计算机视觉任务上取得了优异的性能，例如，在与其他模型尺寸相当、使用相同训练策略时，RMT在ImageNet-1k上实现了84.1%的Top1准确率，这是首次出现的最高准确率。

**3. 核心关键词**

- `Retentive Network` (`保留网络`)

- `Transformer` (`变压器`)

- `Vision Model` (`视觉模型`)

- `Spatial Distance` (`空间距离`)

- `Decomposition` (`分解`)

**4. 评价**

- **实用性：5分**  

从实验结果来看，RMT在各种计算机视觉任务，包括对象检测、实例分割和语义分割等方面都表现出色，因此具有较高的实用性。

- **创新性：5分**  

本文首次将RetNet应用到计算机视觉中，并进行了创新性的建模分解，具有较高的创新性。

- **推荐度：5分**   

作者对计算机视觉领域的贡献巨大，不仅提出一种减少全局建模计算成本的新方法，而且其模型在各项任务上的性能都优于现有的视觉主干网络，非常值得推荐。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.11523)

## A Paradigm Shift in Machine Translation: Boosting Translation  Performance of Large Language Models

1. **介绍本文的主要工作**

   本文提出了一种创新的、专为机器翻译任务设计的大型语言模型(LLM)微调方法，以提高模型的翻译性能。所提出的方法包含两个微调阶段：首先在单语数据上进行初始微调，然后在一小部分高质量平行数据上进行后续微调。作者将通过这种策略开发的LLM称为高级语言模型翻译器 (ALMA)。实验结果表明，该模型在WMT'21和WMT'22测试数据集中的10种翻译方向上，相较于零射击性能，平均能提升超过12 BLEU和12 COMET，性能明显优于所有先前的工作，甚至优于参数量更大的NLLB-54B模型和GPT-3.5-text-davinci-003。

   

2. **本文工作的主要亮点**

   本文的主要亮点在于提出了一个新的LLM微调方法，有效地提升了模型在翻译任务上的性能，使得中等规模的LLM能与传统的有监督编码器-解码器翻译模型相媲美，并且不需要依赖大量平行数据，这无疑为机器翻译的新的训练范例奠定了基础。

3. **核心关键词**

    - Generative Large Language Models (生成大型语言模型)

    - Machine Translation (机器翻译)

    - Fine-Tuning Strategy (微调策略)

    - Monolingual and Parallel Data (单语和平行数据)

    - Advanced Language Model-based trAnslator (ALMA, 高级语言模型翻译器)

4. **打分**

    - 实用性：4.5分。方法不依赖大量平行数据，方便实施，且提升明显，实用性强。

    - 创新性：4.5分。提出的微调方法针对性强，为机器翻译的模型训练范例带来新的视角。

    - 推荐度：5分。鉴于本研究在机器翻译任务上取得了显著的效果提升，且实施便利，值得推荐参考。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.11674)

## LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset

**1. 本文主要工作**

这篇论文介绍了一个名为LMSYS-Chat-1M的大规模数据集，该数据集包含了一百万条来自25个顶级大规模语言模型（LLMs）的真实世界对话。这些数据是从Vicuna demo和Chatbot Arena网站上的21万个独立IP地址中获取的。文章提供了数据集内容的概述，包括构建过程、基本统计信息和主题分布，并深入挖掘其多样性、独创性和规模，展现了数据集的多种用途。

**2. 主要亮点**

- 数据集规模巨大，包含真实世界对话一百万条，是由25个顶级LLMs生成，从21万个独立IP地址中获取，保证了样本多样性和真实性。

- 数据集经过精心策划和统计，数据质量高。

- 数据集的用途多种多样，体现了其通用性，例如：开发内容调节模型，建立安全基准，训练跟随指令的模型，以及创造具有挑战性的基准问题。

**3. 核心关键词**

- LLM（大规模语言模型）

- LMSYS-Chat-1M（LMSYS-Chat-1M数据集）

- Content Moderation（内容调节）

- Safety Benchmark（安全基准）

- Instruction-following Models（指令跟随模型）

**4. 各项评分**

- 实用性：★★★★★ （5分，数据集广泛适用于许多研究领域，能被用于开发新的模型或优化现有模型）

- 创新性：★★★★☆ （4分，数据集提供了许多功用，能够用于研究各种复杂的任务，但仍然在现有的研究范式下进行）

- 推荐度：★★★★☆ （4分，由于其高质量和多样性，该数据集对任何研究LLMs的人来说都是一个极好的资源）

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.11998)

## LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language  Model as an Agent

1. **本文的主要工作**

   本文提出了一种全新的、无需标记训练数据的零样本、开放词汇的大型语言模型（LLM）基础的3D视觉定位流程——LLM-Grounder。该方法利用LLM将复杂的自然语言查询分解为语义成分，并使用视觉定位工具（如OpenScene或LERF）来识别3D场景中的对象。然后，LLM评估提出的对象之间的空间和常识关系，做出最终的定位决策。该方法能够推广到新的3D场景和任意文本查询。

2. **本文工作的主要亮点**

   主要亮点在于，LLM-Grounder不需要任何标记训练数据就能实现零样本、开放词汇的3D视觉定位，表现出了良好的泛化能力。此外，通过在ScanRefer基准上的评估，LLM-Grounder展示出了最新的零样本定位准确率，证明了大型语言模型能够显著提高定位能力，尤其是对于复杂语言查询。

3. **核心关键词**

   - `Large Language Model` (大型语言模型)

   - `Open-Vocabulary 3D Visual Grounding` (开放词汇3D视觉定位)

   - `Zero-Shot Learning` (零样本学习)

   - `ScanRefer benchmark` (ScanRefer基准)

   - `Robotics` (机器人学)

4. **实用性、创新性和推荐度打分**

   - 实用性：4.5分。LLM-Grounder对于开放词汇、3D视觉定位的处理具有显著的影响，十分重要也相当实用，允许家庭机器人进行导航，操纵对象，甚至根据其环境回答问题。

   - 创新性：5分。这是一种全新的，利用大型语言模型实现开放词汇的3D视觉定位的方法，不需依赖标记训练数据，开创了新的可能性。

   - 推荐度：5分。该论文的强大表现和在ScanRefer基准上的最新结果表明了该模型在实践中的潜力，因此我强烈推荐阅读这篇论文。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.12311)

## MetaMath: Bootstrap Your Own Mathematical Questions for Large Language  Models

### 本文主要工作

本文主要提出了一个名为MetaMath的经过微调的语言模型，专门用于数学推理。首先，通过重写问题并从多个角度进行观察，而不需要额外知识，来自举数学问题。重塑过程最终产生了一个新的数据集——MetaMathQA。然后在MetaMathQA上微调LLaMA-2模型。在两个数学推理的流行基准（GSM8K和MATH）上的实验结果表明，MetaMath在很大程度上优于大部分开源的大型语言模型(LLMs)。并且，文章公开提供了MetaMathQA数据集，各种大小的MetaMath模型以及训练代码。

### 主要亮点

- 提出了一个新的微调语言模型MetaMath，专门针对数学推理。

- 提出了一个新型的问题引导生成技术，无需额外知识，就能从多个角度重写问题，以形成MetaMathQA数据集。

- 在GSM8K和MATH两个数学推理基准上，MetaMath比多数开源的大语言模型性能优越。

- MetaMath-7B模型在GSM8K和MATH上的准确率分别达到了66.4%和19.4%，超过了同等大小的最先进模型11.5%和8.7%。

- 提供了公开的MetaMathQA数据集，不同模型大小的MetaMath模型以及训练代码。

### 核心关键词

- `MetaMath` (`元数学`)

- `LLMs` (`大型语言模型`)

- `MetaMathQA` (`元数学问题答案`)

- `Mathematical reasoning` (`数学推理`)

- `Fine-tuning` (`微调`)

### 评分

- 实用性：4.5分。MetaMath具有显著的数学问题解决能力，具有很强的实用性，便于在各种数学相关领域内使用。

- 创新性：4分。本文通过引导生成数学问题并在此基础上训练语言模型，这种方法在处理复杂的数学推理问题上显示出了创新性。

- 推荐度：4.5分。鉴于MetaMath在短期内解决复杂数学问题的能力，以及数据集和模型的公开可用性，强烈推荐相关领域的研究者和实践者阅读和使用。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.12284)

## BTLM-3B-8K: 7B Parameter Performance in a 3B Parameter Model

### 1. 主要工作

本文主要介绍了一种名为"BTLM-3B-8K"的新型语言模型。该模型具有30亿参数，通过在SlimPajama数据集上进行训练，BTLM-3B-8K在所有类型的下游任务中皆超越了现有的所有30亿参数模型。实际上，BTLM-3B-8K甚至可以与某些70亿参数模型竞争。此外，相较于其他模型，该模型在处理长文本上的表现更为出色。

### 2. 主要亮点

模型的主要亮点在于：

- BTLM-3B-8K在实现与70亿参数模型相抗衡的性能时仅需要30亿的参数。

- 在处理长文本(最长8192个上下文长度)时表现优异，超越了MPT-7B-8K和XGen-7B-8K等模型。

- 通过压缩，模型仅需3GB内存和4位精度，比70亿参数模型少了2.5倍的推理计算。这有助于在移动设备和边缘设备上开放强大的语言模型。

### 3. 核心关键词

- BTLM-3B-8K (`BTLM-3B-8K`)

- SlimPajama dataset (`SlimPajama数据集`)

- ALiBi position embeddings (`ALiBi位置嵌入`)

- SwiGLU nonlinearity (`SwiGLU非线性`)

- 4-bit precision (`4位精度`)

### 4. 评分

- 实用性：5分。该模型具有较低的参数和较高的性能，可用于各种类型的NLP任务，且易于部署在移动设备和边缘设备上。

- 创新性：4分。新模型引入了几个创新元素，如ALiBi位置嵌入和SwiGLU非线性，这有助于提升模型的表现，具有一定的创新价值。

- 推荐度：4.5分。考虑到其在下游任务中的卓越性能和在处理长文本中的效率，我非常推荐此模型。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.11568)

## Boolformer: Symbolic Regression of Logic Functions with Transformers

**1. 本文的主要工作**

本文介绍了一种名为Boolformer的Transformers架构，这是第一种被训练以执行布尔函数端到端符号回归的Transformers架构。首先，作者展示了在提供干净的真值表时，该模型能预测复杂功能的紧凑公式。然后，证明了当提供不完整和嘈杂的观察时，Boolformer具有找到近似表达式的能力。在一系列真实世界的二元分类数据集上评价了Boolformer，并将其应用于模拟基因调控网络动态的广泛任务。

**2. 本文工作的主要亮点**

Boolformer在预测复杂函数的能力，以及在不完整和嘈杂的观测中找到近似表达式的能力上展示了其独特之处。此外，与现有的基因算法相比，它的速度提升了几个数量级，且具有高度的可解释性，有潜力成为经典机器学习方法的可解释替代品。

**3. 核心关键词**

- `Boolformer` (Boolformer)

- `Transformers architecture` (Transformers架构)

- `Symbolic Regression` (符号回归)

- `Boolean Functions` (布尔函数)

- `Truth Table` (真值表)

**4. 评分**

- 实用性评分：4分。该模型具有解决现实问题的潜力，特别是在处理布尔函数回归任务和基因调控网络模型方面，有很高的实用性。

- 创新性评分：5分。这是首个被训练以执行布尔函数的端到端符号回归的Transformers架构，具有很高的创新性。

- 推荐度评分：4分。Boolformer灵活且高效，实用性和创新性都很强，对于研究人员及工程师具有很高的推荐价值。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.12207)