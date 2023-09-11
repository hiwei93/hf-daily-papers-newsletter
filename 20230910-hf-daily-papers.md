# 【2023-09-10】Huggingface 每日论文速览

- [Huggingface Daily Papers 2023-09-10](https://huggingface.co/papers?date=2023-09-10) 共推荐 13 篇论文。

> 💡说明：
>
> - 本文对 Huggingface Daily Papers 推荐的论文从：主要工作、主要两点、关键词和评估四个方面进行速览。
>
> - 论文的速览内容基于论文的摘要，使用 GPT-4 进行内容生成，然后使用程序将内容整合，并以 Markdown 文本呈现。
## Large Language Models as Optimizers

1. **介绍本文的主要工作**

    本文提出了一种名为OPRO (Optimization by PROmpting) 的方法，利用大型语言模型（LLMs）作为优化器来解决优化难题。工作首先在线性回归和旅行商问题上进行实验，然后进行提示优化，目标是找到能最大化任务准确性的指令。

2. **本文工作的主要亮点**

    本文的主要亮点在于使用大型语言模型作为优化器，此方法的独特之处在于它将优化任务以自然语言描述，并使用前一步生成的解决方案作为提醒来生成新的解决方案。最后，这些新解决方案被评估并添加到下一步优化的提示中。此外，通过各种大型语言模型的试验，论文显示OPRO优化的最佳提示在GSM8K上超越了人类设计的提示8%，在Big-Bench Hard任务上超越了50%。

3. **核心关键词**

    - `Large Language Models` (`大型语言模型`)

    - `Optimization` (`优化`)

    - `Prompting` (`提示`)

    - `Linear Regression` (`线性回归`)

    - `Traveling Salesman Problem` (`旅行商问题`)

4. **打分**

    - 实用性：4.5/5

    - 创新性：5/5

    - 推荐度：4.5/5

总的来说，本文展示了将大型语言模型用作优化器的强大潜力，可能对许多实际应用产生积极的影响，尤其是在梯度难以获取的情况下。此外，该方法在创新性上表现出色。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03409)
## FLM-101B: An Open LLM and How to Train It with $100K Budget

1. 本文主要工作：

   

   这篇论文提出了一种在受限预算下有效训练大规模语言模型（Large Language Models，LLMs）的策略。该策略允许在仅使用100K预算的情况下训练一个拥有101B参数和0.31TB tokens的LLM。研究者也引入了一套全面的评估范式来公正客观地评估LLMs，以补充现有更侧重于知识导向能力的评估。此外，具有创新性的是，他们还开发了一个新的性能评估基准。

2. 亮点：

   - 开发出了一种在预算限制下训练大规模语言模型的有效策略。

   - 设立了一个全新的评估基准以全面、公正地评价LLM的性能。

   - 成功训练出了一个具有较高性能的新模型FLM-101B，并将其开源。

3. 核心关键词：`Large Language Model` (大规模语言模型), `Cost-effective Training` (经济有效的训练), `Evaluation Paradigm` (评估范式), `Intelligence Benchmark` (智能基准), `Open-source Model` (开源模型)

4. 打分：

   - 实用性：5/5. 该论文解决了训练大规模语言模型高计算成本的问题，具有非常高的实用性。

   - 创新性：4/5. 论文在经济有效的训练策略和评估范式 方面做出了创新，但在理论上的创新尚有可提升的空间。

   - 推荐度：5/5. 推荐阅读此篇文章，因为它不仅对资源有限的研究者提供了新的训练策略，而且还提供了一种新的评估方式，对于LLM的研究和应用有很大的帮助。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03852)
## Tracking Anything with Decoupled Video Segmentation

1. **介绍本文的主要工作**

   本文提出一种名为"Decoupled Video Segmentation Approach"（DEVA）的方法来进行视频分割。DEVA由特定任务的图像级分割和类别/任务无关的双向时间传播组成。这种设计无需对每个独立任务进行视频数据的训练，只需要针对目标任务的图像级模型（更便宜的训练）和一次性训练的普适的时间传播模型。

2. **本文工作的主要亮点**

   DEVA利用双向传播完成不同帧的分割假设的（半）联机融合，生成连贯的分割结果。相较其他端到端方法，在多个数据稀缺的任务中，如大词汇量视频全景分割、开放世界视频分割、指代视频分割和无监督视频对象分割等，DEVA表现出了优越的性能。

3. **核心关键词**

   

   - `Decoupled Video Segmentation Approach` (解耦的视频分割方法)

   - `Image-level segmentation` (图像级分割)

   - `Bi-directional temporal propagation` (双向时间传播)

   - `Data-scarce tasks` (数据稀缺任务)

   - `Online fusion` (在线融合)

4. **从实用性、创新性和推荐度进行打分**

   

   - 实用性：4分

   - 创新性：5分

   - 推荐度：4分

   *注：分数基于该方法在处理数据稀缺任务中的优越性质、解决数据稀缺问题的创新策略，以及对于未来的视频分割任务具有一定的应用价值和推广潜力。*

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03903)
## GPT Can Solve Mathematical Problems Without a Calculator

1. **介绍本文的主要工作**

    本文主要挑战了大型语言模型无法准确完成算术操作特别是涉及大于8位数的乘法、小数和分数操作的常见观念。作者通过大量训练数据，展示了一个2亿参数的语言模型可以准确执行多位数算术操作，并且没有数据泄漏。同时，该模型显著优于GPT-4的多位数乘法精度（只有4.3%）。 作者还对MathGLM进行了微调，使用更多的多步骤算术操作和文字描述的数学问题，从而在5000个样本的中文数学问题测试集上达到了与GPT-4类似的表现。

2. **本文工作的主要亮点**

    - 挑战了大型语言模型不能准确执行算术操作的普遍认识。

    - 利用大量训练数据，让一款2亿参数的模型能够正确执行有关大数乘法、小数和分数操作的任务。

    - 提出的模型在多位数乘法精度上显著领先于GPT-4。

    - 在5000个样本的中文数学问题测试集上达到了与GPT-4类似的性能。

3. **核心关键词**

    - `Large Language Model` (大型语言模型)

    - `Mathematics` (数学)

    - `Arithmetic Operations` (算术运算)

    - `Data Leakage` (数据泄漏)

    - `Fine Tuning` (微调)

    

4. **打分**

    - **实用性**：4.5

    - **创新性**：4.0

    - **推荐度**：4.0

    提出的语言模型具有较强的实用性，能解决数学问题，对教育等许多领域都有一定的参考价值。研究给既有的认识提供了挑战，展示了语言模型在算术操作的潜力，具有较高的创新性。这篇文章值得推荐给对人工智能和数学教育的研究人员。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03241)
## ProPainter: Improving Propagation and Transformer for Video Inpainting

1. **本文主要工作**

  本文提出了一个改进的视频修复（Video Inpainting）框架ProPainter。该框架涉及强化的传播和高效的Transformer，特别是引入了双域传播，结合了图像和特征扭曲的优势，可靠地利用全局对应关系。此外，还提出了一个掩码引导的稀疏视频Transformer，通过丢弃不必要和冗余的标记来实现高效。

2. **本文工作的主要亮点**

  文章的亮点在于创新的ProPainter框架，其通过引入双域传播和掩码引导的稀疏视频Transformer，解决了传统方法在空间错位和跨帧信息获取上的问题。这使得ProPainter在PSNR中超出先前的方法1.46 dB，同时保持了良好的效率。

3. **核心关键词**

    - `ProPainter` (ProPainter)

    - `Video Inpainting` (视频修复)

    - `Dual-domain Propagation` (双域传播)

    - `Transformer` (Transformer)

    - `Sparse Video Transformer` (稀疏视频Transformer)

4. **评分**

    - **实用性**：4.5分。ProPainter框架在视频修复领域有很强的实用性，能够提高修复质量和效率。

    - **创新性**：4分。本文提出的双域传播和掩码引导的稀疏视频Transformer，是对当前方法的创新改进。

    - **推荐度**：4分。本文研究内容具有较强的学术价值和实用价值，值得在相关领域

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03897)
## ImageBind-LLM: Multi-modality Instruction Tuning

1. **介绍本文的主要工作**

   本文介绍了ImageBind-LLM，一个通过ImageBind对大型语言模型（Large Language Models，LLM）进行多模态指令调整的方法。与现有的主要关注语言和图像指令调整的工作不同，ImageBind-LLM可以响应多模态条件，包括音频、3D点云、视频以及他们的嵌入空间算法，这都是通过仅在图像-文本对齐训练中实现的。

2. **本文工作的主要亮点**

   主要的亮点是利用可学习的绑定网络在LLaMA和ImageBind的图像编码器之间对嵌入空间进行对齐。此外，它通过无注意力且初始化为零的门控机制，在LLaMA的所有层中逐步注入视觉指令。在推理阶段，多模态输入被输入到对应的ImageBind编码器，并由提出的视觉缓存模型处理以进一步提高跨模态嵌入性能。显然，ImageBind-LLM能够对多种模态的指令做出反应，并展示出显著的语言生成质量。

3. **核心关键词**

   - Large Language Models (大型语言模型)

   -  ImageBind (图像绑定)

   -  Multi-modality (多模态)

   -  Embedding Space Alignment (嵌入空间对齐)

   -  Visual Instructions Injection (视觉指令注入)

4. **打分**

   - 实用性：4.5分

   - 创新性：4.8分

   - 推荐度：4.7分

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03905)
## InstructDiffusion: A Generalist Modeling Interface for Vision Tasks

**本文主要工作：**

本文提出了InstructDiffusion，一个用于视觉任务的统一且通用的框架，可以将各种视觉任务对齐到人类指令之下，不需要集成先验知识和预定义每个视觉任务的输出空间，例如类别和坐标。该模型基于扩散过程并被训练用来预测用户指令下的像素。

**本文工作的主要亮点：**

InstructDiffusion可以处理各种视觉任务，包括理解任务（如分割和关键点检测）和生成任务（如编辑和增强）。它甚至能够处理未见过的任务，并在新的数据集上超越了先前的方法。这代表了朝通用建模接口的重要一步，推动了计算机视觉领域的人工智能发展。

**核心关键词：**

- InstructDiffusion (`指导扩散`)

- Diffusion process (`扩散过程`)

- Image-manipulating process (`图像操作过程`)

- Segmentation (`图像分割`)

- Keypoint detection (`关键点检测`)

**评分：**

- **实用性**：4/5，InstructDiffusion可以广泛的应用于各类视觉任务，实用性强。

- **创新性**：5/5，本文将视觉任务与人类指令相对齐是一项重大的创新，甚至能够处理未见过的任务。

- **推荐度**：4.5/5，对于此领域的研究者和工程师，这项工作推动了计算机视觉的人工智能发展，并提供了新的研究方向和实践应用，值得推荐学习。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03895)
## DoLa: Decoding by Contrasting Layers Improves Factuality in Large  Language Models

**1. 文章主要工作**

 这篇论文提出了一种简单的解码策略，用于减少预训练大型语言模型 (LLMs) 中的幻觉生成（即偏离预训练时看到的事实的内容生成）。他们的方法通过对比从后层向词汇空间投影获得的 logits 与早期层的差异，来获得下一个 token 的分布，以此利用 LLMs 中局部化到特定 transformer 层的事实知识。这种对比层次解码（DoLa）方法能有效地提取事实知识，减少错误事实的生成。

**2. 文章亮点**

 DoLa方法能够改善LLM的真实性，降低“误导性”信息的生成。例如，它在TruthfulQA上改善LLaMA家族模型的表现，绝对得分提升12-17%，这体现了它强大的能力，使LLM可靠地生成真实的事实。

**3. 核心关键词**

- Large Language Models (大型语言模型)

- Decoding Strategy (解码策略)

- Logits (逻辑函数)

- Transformer Layers (Transformer 层)

- TruthfulQA (真实性QA)

**4.评分**

- 实用性：5/5，该方法改进了大型语言模型的真实性，有助于提升模型的有效性和可信度。

- 创新性：4.5/5，该文使用一种新的解码策略，通过对比不同层次获取的逻辑函数来改善模型的表现，十分创新。

- 推荐度：5/5，对于大型语言模型的改进十分必要，这篇论文提供了一个有效的改进策略，对这个领域的研究者来说是值得一读的文章。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03883)
## SyncDreamer: Generating Multiview-consistent Images from a Single-view  Image

1. **主要工作**：

本文提出一种名为SyncDreamer的新型扩散模型，它能从单视图图像生成多视图一致的图像。为了处理生成图像在几何和颜色上保持一致性的挑战，我们设计了一个同步多视图扩散模型来模拟多视图图像的联合概率分布， 这使得在单个逆向过程中可以生成多视图一致的图像。

2. **主要亮点**：

SyncDreamer通过一个了解3D的特征注意机制在每个逆向过程的步骤中同步所有生成图像的中间状态，该机制能跨越不同视图连接相应的特征。实验表明，这种模型可以生成具有高一致性的跨不同视图的图像，使其非常适合于各种3D生成任务。

3. **核心关键词**：

 - `SyncDreamer` (`SyncDreamer`)

 - `Diffusion Model` (`扩散模型`)

 - `Single-view image` (`单视图图像`)

 - `Multiview images` (`多视图图像`)

 - `3D-aware feature attention mechanism` (`了解3D的特征注意机制`)

4. **评分**：

 - **实用性**：4.5/5，这个模型的实用性很高，因为它能够从单一视点创造出一致的多视角图像，这对于各种3D任务非常有用。

 - **创新性**：5/5，这个模型表示了独特的方法来应对生成图像在几何和颜色上保持一致性的挑战，并使用了了解3D的特征注意机制来同步所有生成的图像，显示了高度的创新性。

 - **推荐度**：4.5/5，推荐读者阅读并了解这个模型，它在解决一致性问题和从单视图生成多视图图像时展现了非常有价值的新思路。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03453)
## XGen-7B Technical Report

### 1. 主要工作

本文发表了XGen系列模型，这是一系列7B参数模型，对最多8K序列长度进行了训练，并对最多1.5T的tokens进行了训练。同时，文章对XGen模型进行了在公开域指令数据上的微调，创建了其指令调谐的对应模型(XGen-Inst)。文中还对这类模型进行开源，旨在推动研究进步和商业应用。

### 2. 亮点

本文章的大亮点在于，XGen模型不仅对更长的序列长度进行了训练，但其性能也能够与现有的开源大型语言模型相匹敌甚至优于它们。特别的，面对长序列建模任务，8K序列的模型显示出对2K序列的开源大型语言模型的优点。此外，作者们还提供了开源代码，这对于推动技术的发展和商业应用都具有重要价值。

### 3. 核心关键词

- `Large Language Models` (`大型语言模型`)

- `XGen` (`XGen模型`)

- `Sequence Length` (`序列长度`)

- `Instruction-Tuned` (`指令调谐`)

- `Open-Source` (`开源`)

### 4. 评价

- 实用性：4分

- 创新性：4.5分

- 推荐度：4分

XGen系列模型以其能够处理更长序列长度的能力和公开代码的开放性，展现出很高的实用性。该模型在处理长序列的能力方面展现出创新，也因此获得了高分。总体来看，这是一篇值得推荐阅读的文章。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03450)
## Robotic Table Tennis: A Case Study into a High Speed Learning System

1. **介绍本文的主要工作**

    本研究深入探讨了一个实际的机器人学习系统，该系统在以前的研究中已经展示出能够和人类进行数百次的乒乓球比赛并且能够精确地将球返回到指定的目标。该系统综合了高度优化的感知子系统、高速低延迟的机器人控制器、能够防止实际世界损坏同时也能培训零转移策略的仿真范例，以及能够实现在物理机器人上进行自主训练和评估的实际环境重置。

2. **本文工作的主要亮点**

    论文详尽的描绘了系统的实现细节，并探讨了各种设计决策，这些内容在很多其他论文中并未明确提及。此外，本研究对降低各种延迟源、调整训练与部署分布差异、提高感知系统的稳健性、对策略超参数的敏感度和行为空间选择等诸多关键因素进行了深入的实证研究。

3. **核心关键词**

    - `Robotic Learning System` (机器人学习系统)

    - `Perception Subsystem` (感知子系统)

    - `Zero-shot Transfer` (零次转移)

    - `Latency` (延迟)

    - `Autonomous Training` (自主训练)

4. **从实用性、创新性和推荐度进行打分**

    - 实用性：5分，系统的实战能力强，能进行自主训练和评估，具有很高的实用性。

    - 创新性：5分，成功地将多个前沿技术融合在一起，并通过实证研究支持了其中的关键设计决策。

    - 推荐度：5分，内容丰富，成果显著，对到学术界与产业界都有较高的参考价值。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03315)
## Text2Control3D: Controllable 3D Avatar Generation in Neural Radiance  Fields using Geometry-Guided Text-to-Image Diffusion Model

1. **介绍本文的主要工作**

   本文提出了一个名为Text2Control3D的可控文本至3D头像生成方法。该方法可以根据一段由手持相机随意拍摄的单眼镜头视频，控制头像的面部表情。主要策略是使用ControlNet生成的一组受控的视点感知图像来构造Neural Radiance Fields（NeRF）中的3D头像，并从输入视频中提取深度图作为ControlNet的条件输入。

2. **本文工作的主要亮点**

   导出的3D头像可以根据输入的文本进行表情、外观的控制。将视点无关的纹理问题进行了处理，并考虑了每个图像的几何变化，为不严格几何一致的图像训练NeRF，由此构建出了变形NeRF的规范空间。

3. **核心关键词**

   - `Neural Radiance Fields` (神经辐射场)

   - `ControlNet` (控制网络)

   - `Text-to-3D Generation` (文本到3D生成)

   - `viewpoint-aware images` (视点感知图像)

   - `deformable NeRF` (可变形NeRF)

4. **实用性、创新性和推荐度打分**

   - 实用性：4分，本文提出的方法可以在3D头像构造中应用，有相当多的实用性。

   - 创新性：4分，该工作在文本至3D生成的控制性方面有显著的创新，特别是视点感知图像和可变形NeRF的应用。

   - 推荐度：4分，对于那些在计算机视觉和生成模型领域工作的人来说，这是一个值得阅读的文献。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03550)
## Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation

**1. 介绍本文的主要工作**

本文提出了一个名为"Reuse and Diffuse" (简称 VidRD)的新框架，用于更高效地进行文本到视频的生成。这个框架参考了Latent Diffusion Models（LDMs）在图像合成方面的成功。它通过重复使用原始的潜在特征并逐步引入已生成的视频帧的扩散过程，能够生成更多的视频帧。

**2. 本文工作的主要亮点**

主要亮点包括提出有效解决计算和内存限制的VidRD框架，并优化了像素空间与潜在空间转换的自动编码器，注入了时间层以提高时间一致性。此外，他们还制定了一组策略，将多种现有数据集的内容进行了有效组合，从而获得了更具多样性的视频-文本数据。

**3. 核心关键词**

- `Latent Diffusion Models` (`潜在扩散模型`)

- `Text-to-Video Generation` (`文本到视频生成`)

- `Autoencoder` (`自动编码器`)

- `Temporal Consistency` (`时间一致性`)

- `Data Composition` (`数据组合`)

**4. 实用性、创新性和推荐度**

- **实用性：**4.0/5.0。此方法可以在资源受限的情况下实现更多的文本到视频帧生成，具有一定的实用性。

- **创新性：**4.5/5.0。该论文的“重用和扩散”模型以及时间一致性的自动编码器注入方法都显示出较高的创新性。

- **推荐度：**4.0/5.0。这篇论文对解决文本到视频生成的复杂性和资源问题提出了一种全新的解决路径，推荐给对此领域感兴趣的读者。

[到 Huggingface 论文主页查看详情](https://huggingface.co/papers/2309.03549)