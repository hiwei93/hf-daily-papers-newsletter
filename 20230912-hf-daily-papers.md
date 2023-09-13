# 【2023-09-12】Huggingface 每日论文速览

- [Huggingface Daily Papers 2023-09-12](https://huggingface.co/papers?date=2023-09-12) 共推荐 8 篇论文。

> 💡说明：
>
> - 本文对 Huggingface Daily Papers 推荐的论文从：主要工作、主要两点、关键词和评估四个方面进行速览。
>
> - 论文的速览内容基于论文的摘要，使用 GPT-4 进行内容生成，然后使用程序将内容整合，并以 Markdown 文本呈现。
## Textbooks Are All You Need II: phi-1.5 technical report

**1. 本文主要工作**

本文基于“Textbooks Are All You Need”的理念，继续研究了小型Transformer-based语言模型的能力，重点研究自然语言中的常识推理。通过使用现有的大型语言模型（LLMs）生成“教科书级别”的数据进行增强学习，相比于传统的网络数据，本文提出了一个新的1.3亿参数模型phi-1.5。该模型在自然语言任务上的表现可与5倍大小的模型比拟，且在更复杂的推理任务（如小学数学和基本编程）上优于大多数非前沿LLMs。

**2. 主要亮点**

首先，本文的phi-1.5模型在保持小型化的同时，实现了极高的性能，能够与大五倍的模型相抗衡，这在模型设计领域具有重要意义。其次，除了较具挑战性的任务，如小学数学和基本编码之外，phi-1.5模型也展现了较大的LLMs的许多特性，如“一步一步思考”或进行一些初级的情境学习。最后，由于没有使用网络数据，该模型在减少产生有害和偏见的生成方面也取得了进步。

**3. 核心关键词**

- Transformer-based models（基于Transformer的模型）

- Large Language Models (LLMs)（大型语言模型）

- Textbook Quality Data（教科书级别的数据）

- Common Sense Reasoning（常识推理）

- In-context Learning（情境学习）

**4. 打分**

- 实用性：4/5

- 创新性：5/5

- 推荐度：4.5/5

本文所提出的模型在自然语言处理和推理任务上展现出了较高的性能，实用性较强。同时，模型的小型化、利用大型语言模型生成教科书级别的数据进行增强学习的策略，以及在没有网络数据的情况下减少有害、偏见生成的设计，均具有较高的创新性。对于关注人工智能、机器学习、自然处理等领域的研究者和开发者，本文都值得一读。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.05463)
## NExT-GPT: Any-to-Any Multimodal LLM

**1. 介绍本文的主要工作**

本文提出了一个端对端的通用的任意模态的大型语言模型 (LLM)系统——NExT-GPT。通过连接LLM和多模态适配器以及不同的扩散解码器，NExT-GPT能够在任意组合的文本、图像、视频和音频中理解输入并产生输出。同时还引入了模态切换指令调整 (MosIT) 并手动策划了一个高质量的MosIT数据集，使NExT-GPT具有跨模态语义理解和内容生成的能力。

**2. 本文工作的主要亮点**

1. 首创提出任意到任意的多模态大型语言模型系统，能够理解并产生各类模态的数据。

2. 通过模态切换指令调整和高质量数据集的使用，赋予模型复杂的跨模态语义理解和内容生成能力。

**3. 核心关键词**

- Large Language Models (LLM) (大型语言模型)

- Multimodal Adaptors (多模态适配器)

- Diffusion Decoders (扩散解码器)

- MosIT (模态切换指令调整)

- Cross-Modal Semantic Understanding (跨模态语义理解)

**4. 从实用性、创新性和推荐度进行打分**

- 实用性 ：4.5/5，该研究开创的模型可以广泛应用于如提高人工智能与人类交互的自然性等各种场景。

- 创新性 ：5/5，此项研究引入了全新的模态切换指令调整，填补了语言模型在处理多模态内容上的空白。

- 推荐度 ：4.5/5，给了高分，因其为进一步发展更人性化的AI研究铺平了

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.05519)
## When Less is More: Investigating Data Pruning for Pretraining LLMs at  Scale

### 论文总结

#### 1. 本文的主要工作

这篇文章探讨了数据质量评估指标对于大语言模型 (LLMs) 预训练数据集的修剪过程。作者用一组度量来评价数据质量，包括困惑度（perplexity）等简单评价指标和其他计算密集型的评价指标。然后在修剪后的数据集上对 LLMS 进行训练，并比较了结果。结果显示，效果最好的是最简单的困惑度方法，甚至在只使用原始训练数据集30%的情况下仍能改善原始模型。

#### 2. 本文工作的主要亮点

主要亮点在于，该研究发现，通过评估预训练数据的质量并相应地修剪数据，我们甚至能在用较少的数据进行训练时仍保持性能不变。这主要是通过使用简单的困惑度指标实现的，这与传统的复杂和计算密集型的质量评估方法相比，表现出很大的优势。

#### 3. 核心关键词

- `Large Language Models` (大语言模型)

- `Data Pruning` (数据修剪)

- `Perplexity` (困惑度)

- `Data Quality Estimation` (数据质量评估)

- `Corpora Curation` (语料库策划)

#### 4. 打分

- 实用性：4分。通过修剪预训练数据集，能用更少的数据获得相等甚至更好的结果。这可以在实际应用中节省存储和计算资源。

- 创新性：4分：尽管数据修剪技术往往在许多领域中都被用到，但在大语言模型的训练中并不常见。这篇文章以一种新颖的方式探索了数据修剪在这一领域中的应用。

- 推荐度：4分。本文的研究结果为自动策划高质量语料库提供了新的策略，这对于大语言模型的发展具

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.04564)
## Neurons in Large Language Models: Dead, N-gram, Positional

**1. 本文的主要工作**

本文对大规模语言模型进行了轻量级的分析，主要针对125m到66b参数的OPT模型系列。分析的重点在于，一个FFN神经元是否被激活。通过对大量不同数据下的模型激活情况进行调查，作者发现模型中的许多神经元实际上是"死亡"的，即他们在处理这些数据时并未被启动，占据了模型参数的相当一部分。此外，作者还发现了模型中存在的专门处理离散特性的有效神经元，以及随着模型规模的增长，这些神经元变得越来越多，提供了更多信息。

**2. 本文工作的主要亮点**

* 对大规模语言模型的轻量级分析，易于实施且具有可扩展性。

* 揭示了模型中的一部分神经元在处理大量不同数据时实际上并未被启动，即大部分神经元是"死亡"的。这一新发现可能挑战了以往对模型的理解和使用方式。 

* 发现了模型中专门处理离散特性和令牌的有效神经元，进一步在某种程度上解释了模型内部的工作机制。

**3. 核心关键词**

* `Large Language Models` (`大规模语言模型`)

* `OPT Models` (`OPT模型`)

* `Dead Neurons` (`死亡神经元`)

* `N-gram Detectors` (`n元词组探测器`)

* `Positional Neurons` (`位置神经元`)

**4. 打分**

* 实用性：4分。找出模型中可能存在的冗余神经元并识别其有效神经元，这对模型的优化和应用有一定的指导意义。然而，在具体的实用方法上，文章的描述尚不明确。

* 创新性：5分。本文提出的大语言模型分析方法，尤其是"死亡神经元"的观察和解释，都表现出较高的创新性。首次揭示了模型信息流中的消除（而非添加）机制，对理解语言模型提供了新的视角。

* 推荐度：4分。对于研究人员，本文具有较高的参考价值，提供了新的视角和方法分析语言模型。但对于普通读者，可能需要在理解大规模模型和神经元概念方面投入更多的精力。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.04827)
## MADLAD-400: A Multilingual And Document-Level Large Audited Dataset

### 1. 主要工作

本文介绍了名为MADLAD-400的手工审核数据集，这是一个基于CommonCrawl的通用领域3T token单语数据集，覆盖了419种语言。作者讨论了通过自我审核MADLAD-400揭示的限制，以及数据审核在数据集创建过程中的作用。然后，作者在公开可用的2500亿token上训练并发布了一个达到10.7B参数的多语言机器翻译模型，覆盖了450多种语言，结果表明它能与更大的模型竞争。此外，作者还训练了一个8B参数的语言模型，并评估了其在少量样本翻译上的结果。这些基线模型都已被允许研究社区使用。

### 2. 主要亮点

这篇文章的主要亮点在于它不仅创建了一个包含多种语言的、广泛覆盖的数据集MADLAD-400，而且还通过自我审核揭示出了数据集的限制。这篇文章也开发了一个在超过450种语言的大规模数据上训练的传递模型，证明了它与大规模模型的竞争力。除此之外，实现了少量样本翻译的语言模型同样也是一个亮点。

### 3. 核心关键词

- MADLAD-400（MADLAD-400）

- Multilingual machine translation model（多语言机器翻译模型）

- Language model（语言模型）

- Data auditing（数据审核）

- Few-shot translation（少量样本翻译）

### 4. 评分

实用性：4.5/5  

MADLAD-400数据集与多语言翻译模型在现实场景中具有极高的实用性，特别是对包括但不限于机器翻译、自然语言处理的研究和应用。

创新性：5/5  

创建了一个包括了419种语言的数据集，并且使用大规模数据进行训练，发表了涵盖了450种语言的多语言翻译模型，这在目前还是非常新颖的。

推荐度：5/5  

尽管这是一个高度专业的研究，对于NLP领域的研究者和从业者来说，它非常值得阅读。作者为研究社区提供了多语言翻译模型和语言模型基线，可以帮助推动该领域的发展。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.04662)
## Optimize Weight Rounding via Signed Gradient Descent for the  Quantization of LLMs

**1. 介绍本文的主要工作**

本论文针对大型语言模型 (LLMs) 部署过程中因其存储和内存要求高而带来的挑战，提出了一种优化权重取整的有效方法。论文主要研究的是3和4位的权重量化，在位数减少的情况下，量化网格扩大，使得向上和向下取整的重要性提高。本研究提出了一种精简而高效的权重优化取整任务的方法，名为 SignRound，该方法使用带符号的梯度下降进行轻量级的块级调优，在400步内可获得优秀的结果。

**2. 本文工作的主要亮点**

本文的主要亮点在于提出了一种新的权重取整优化方法 - SignRound，使用带符号的梯度下降进行轻量级的块级调优，短短400步内就能获得显著的成果。SignRound不仅表现出了优于现有的最近取整 (RTN) 方案的性能，而且与近期的其他方法相比具有显著的竞争力，且无需添加额外的推理开销。

**3. 核心关键词**

- Large Language Models (大型语言模型)

- Weight-only quantization (仅权重量化)

- Rounding (取整)

- SignRound (符号圆)

- Signed Gradient Descent (带符号梯度下降)

**4. 从实用性、创新性和推荐度进行打分**

- 实用性：4.5/5。在大型语言模型的部署时，优化存储和内存需求是一个重要且挑战性的问题，SignRound 方法应对这个问题提供了有效的解决方案。

- 创新性：4.5/5。该论文中提出的 SignRound 权重优化取整方法是一种新的，且无需添加额外的推理开销的解决方案，这展现了显著的创新性。

- 推荐度：5/5。文中所提出的 SignRound 方法有很强的实用性且充满创新，对于研究和实践人员具有很高的参考价值，因此推荐读者阅读。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.05516)
## FIAT: Fusing learning paradigms with Instruction-Accelerated Tuning

### 文章工作概述

本论文详细研究了当前大型语言模型（LLMs）的学习范式——上下文学习（ICL）和全面微调，它们各自基于可用的数据、模型规模、计算成本、易用性以及最终质量的衡量标准有着各自的优势和限制。对这两种学习范式进行独特的描述，并按照它们的自然关联提出全新的学习范式FIAT，该范式将这两种范式的优点结合在一起，使得可以使用最大的模型进行注释引导指令以及逻辑推理，同时，还可以采用类似的方法对具有参数效率的中等规模LLM进行参数更新。对FIAT在多元语言任务中的效果进行评估，结果表明，FIAT在从100-10000示例的规模范围内，表现均优于ICL和全面微调。

### 主要亮点

- 强调了ICL和全面微调的自然联系，并对其进行了独特的描述。

- 提出了全新学习范式FIAT，这一范式将ICL和全面微调的优点结合在一起。

- 对FIAT在多元语言任务中的效果进行评估，成功地证明了其在各种规模下的优越性。

### 核心关键词

- `Large language models` (`大型语言模型`)

- `In-context Learning` (`在上下文中学习`)

- `Fine-tuning` (`微调`)

- `FIAT (Fusing Instruction-Accelerated Tuning)` (`FIAT（融合指令加速调整）`)

- `Parameter updates` (`参数更新`)

### 打分

- 实用性：4 分

- 创新性：5 分

- 推荐度：4 分

总体来看，本文提出了一个新的学习范式FIAT，将ICL和全面微调的优点结合在一起，其实用性和创新性非常出色。在多元语言任务中的表现结果表明，该范式在不同规模的任务上都有优于ICL和全面微调的表现，这进一步提高了其实用性和推荐度。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.04663)
## Dynamic Mesh-Aware Radiance Fields

**论文总结：**

1. **介绍本文的主要工作**

   本论文探讨了如何在具有物理一致性的NeRF（Neural Radiance Fields）场景中嵌入多边形网格资产（即模型），以便在渲染和动态模拟中被使用。通过在渲染和模拟中实现网格与NeRF的双向联接，研究者设计了一种更新直射射线上的亮度和吞吐量的有效算法，并考虑如何将该混合表面-体积公式有效地集成到一个高性能物理模拟器中，支持布料、刚性和软体。

2. **本文工作的主要亮点**

   本研究首次设计了一种在渲染和模拟中有效结合网格与NeRF的方法。同时，考虑到路径跟踪器假定的线性颜色空间与标准NeRF使用的sRGB颜色空间存在差异，研究者特意利用高动态范围（HDR）图像来训练NeRF。此外，他们还提出了一种估计光源并对NeRF投影阴影的策略。

3. **核心关键词**

   

   - `Neural Radiance Fields (NeRF)` (`神经辐射场`)

   - `Mesh-NeRF Coupling` (`网格与NeRF耦合`)

   - `Light Transport Equations` (`光传输方程`)

   - `Rendering And Simulation` (`渲染和模拟`)

   - `High Dynamic Range (HDR)` (`高动态范围`)

4. **评分**

   - 实用性：4分

   - 创新性：5分

   - 推荐度：4分

从实用性方面考虑，它提供了一种新的方法，能更有效地在复杂的渲染和模拟任务中使用网格和NeRF。从创新性方面看，这是首次设计了一种在渲染和模拟过程中有效结合网格与NeRF的方法。推荐度方面，对于在计算机图形学和相关领域的研究者们，这篇文章是值得一读的。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.04581)