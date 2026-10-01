---
title: "蒸馏数据真的鲁棒吗？从最小边界到 C²R 的对抗式数据蒸馏"
description: "阅读 ICML 2026 论文 Mind Your Margin and Boundary：从鲁棒 margin 出发，理解 C²R 为什么要优先处理困难对抗样本，并用对比学习拉开决策边界。"
date: 2026-10-01
topics:
  - 数据蒸馏
  - 对抗鲁棒性
  - 论文阅读
tags:
  - Dataset Distillation
  - Robustness
  - Computer Vision
layout: post.njk
permalink: "/posts/robust-distilled-datasets/"
---

> 本文整理的是 ICML 2026 oral 论文 **Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?**，作者为 Muquan Li、Yingyi Ma、Yihong Huang、Hang Gou、Ke Qin、Ming Li、Yuan-Fang Li 和 Tao He。本文采用公开的 arXiv `2605.20606v2` 版本。文章不把摘要逐句翻译，而是追踪一个问题：训练数据被压缩成每类极少量的合成样本后，鲁棒性优化到底应该关注什么？

数据集蒸馏（Dataset Distillation，DD）试图用一个很小的合成集合替代大规模训练集。理想结果不是让合成图像看起来像原始图片，而是让模型只在这些合成样本上训练，也能学到接近使用完整数据时的决策函数。如果还要求模型抵抗输入扰动，问题就变成：合成集合是否保留了足以支撑稳定决策边界的局部结构？

论文给出的主线答案是：**鲁棒错误由 margin 最小的样本主导，因此不能把所有对抗样本平均对待；同时，只让同类 clean/adversarial 特征靠近还不够，还需要把它们与边界另一侧最接近的类别推开。** C²R（Contrastive Curriculum for Robust Dataset Distillation）分别用 Attack-Aware Curriculum（AAC）和 Contrastive Robustness Loss（CRL）实现这两个想法，并用 Line-Search PGD（LS-PGD）降低反复生成对抗样本的成本。

先把结论的边界说清楚：论文理论部分证明的是 robust margin 与 robust hinge 尾部目标之间的关系，实验则显示 C²R 在指定图像分类基准、网络、IPC 和攻击协议下表现更好。它没有证明 C²R 对任意任务或攻击都最优，也没有提供 certified robustness。

## 1. 背景：数据蒸馏为什么会放大鲁棒性问题

设真实训练分布为 $\mathcal D$，要学习的合成集合为

<div class="math-block">
$$
X=\{(x_s,y_s)\}_{s=1}^{N},
\qquad N\ll |\mathcal D|.
$$
</div>

普通 DD 的目标可以写成双层优化：

<div class="math-block">
$$
\min_{X}\;\mathbb E_{(x,y)\sim\mathcal D}
\left[\ell\left(f_{\theta^\star(X)}(x),y\right)\right]
\quad\text{s.t.}\quad
\theta^\star(X)\in\arg\min_{\theta}
\mathbb E_{(x,y)\sim X}\left[\ell(f_\theta(x),y)\right].
$$
</div>

$X$ 是外层要学习的对象，$\theta^\star(X)$ 是模型只在 $X$ 上训练后的参数。内层只看蒸馏集合，外层却要在真实分布上评价，因此合成样本需要以很小的数量承载对模型更新最有用的训练信号。

这也解释了数据蒸馏与普通训练集采样的差别。普通训练集通常有大量冗余样本，模型可以从许多近似例子中逐渐学到类别规律；蒸馏集合只有很少的可学习图像，每张图对训练轨迹的影响都更大。如果这些图只覆盖类内的典型区域，模型可能学到正确的类别中心，却没有学到哪些变化会把样本推向相邻类别。鲁棒蒸馏因此不仅是在小数据上保持 clean accuracy，还要让有限样本携带边界附近的判别信息。

换句话说，蒸馏样本不必复原真实数据分布的每个细节，但它必须保留对目标训练过程关键的结构。一个小数据集若只包含易分类中心点，训练时看起来很稳定，遇到允许范围内的扰动却可能迅速越界；而把边界样本加入训练信号，也不代表所有合成图都应变成极端对抗样本。关键是让合成数据和更新顺序共同覆盖那些决定错误的局部区域。

加入对抗鲁棒性后，训练目标需要回答两个互相关联的问题：允许的扰动会把样本推到哪个类别，以及模型对这段局部路径是否保持正确。给定样本 $(x,y)$ 和扰动集合 $\Delta$，鲁棒训练要考虑集合内所有可能的 $x+\delta$，而不只是随机增强后的一两个点。因此，蒸馏集合需要同时保留类内变化、类别边界和扰动邻域中的决策稳定性。只增加 clean 图像数量，不一定增加了边界附近有用的监督信号。

论文把已有鲁棒 DD 的问题归纳为两个方面。第一，许多方法对所有 adversarial companion 近似均匀处理，但鲁棒错误由少数最危险的低 margin 样本主导，平均信号会把它们淹没。第二，类均值对齐可以让 clean/adversarial 特征整体接近，却没有直接处理某个实例最接近的异类竞争者。这两个问题分别对应 C²R 的 AAC 和 CRL：前者改变“先关注谁”，后者改变“把谁推开”。这一区分很重要：即便一个模型把同类 clean 与 adversarial 特征拉得很近，如果它们仍然贴着另一个类别的特征，分类边界依旧可能很脆弱。

## 2. 理论直觉：为什么最小 robust margin 值得优先处理

### 2.1 从分类 margin 到 robust margin

设分类器输出 $K$ 个 logits，真实标签为 $y$。真实类别与最强竞争类别之间的 logit margin 是

<div class="math-block">
$$
g_\theta(x)=f_y(x)-\max_{k\ne y}f_k(x).
$$
</div>

$g_\theta(x)>0$ 表示真实类别的 logit 高于所有竞争类别；它越接近 0，至少有一个竞争类别就越接近决策边界。这里采用的是 logit 差，而非概率差：softmax 概率会受到其他类别共同归一化的影响，logit margin 则直接比较真实类别与最强对手。margin 的正负决定当前点是否分类正确，大小则给出一个决策优势的代理，但它不是输入空间中以像素距离为单位的几何距离。

给定扰动集合，例如 $\ell_p$ 球

<div class="math-block">
$$
\Delta=\{\delta:\|\delta\|_p\le\varepsilon\},
$$
</div>

robust margin 是这个邻域内最坏的 logit margin：

<div class="math-block">
$$
\underline m(x;\theta)=\min_{\delta\in\Delta}
 g_\theta(x+\delta).
$$
</div>

它回答的是：攻击者在允许范围内，最多能把这个样本的分类优势压低到什么程度？因此，样本在整个扰动集合内保持正确分类，当且仅当 $\underline m(x;\theta)>0$。对应的 robust classification risk 为

<div class="math-block">
$$
\mathcal R_{\mathrm{rob}}(\theta)=
\Pr\left[\underline m(x;\theta)\le 0\right].
$$
</div>

若只看 clean margin，可能会把一个当前分类很自信、但附近存在快速翻转方向的点误判为安全；robust margin 则显式把扰动集合里的最差点纳入定义。这个定义依赖威胁模型 $\Delta$：改变范数或半径，robust margin 和 robust risk 都会改变，所以“鲁棒”必须连同攻击约束一起读。

普通 margin 看当前点离边界多远，robust margin 看攻击者能在邻域中把它推到多靠近边界。后者天然带有最坏情况性质。

直接优化 robust risk 不容易，因此论文采用 robust hinge：

<div class="math-block">
$$
\mathcal L_{\mathrm{hinge}}(\theta)=
\mathbb E_{(x,y)}
\left[1-\underline m(x;\theta)\right]_+,
\qquad [z]_+=\max\{0,z\}.
$$
</div>

当 robust margin 达到 1 时，样本不再贡献 hinge；当 margin 小于 1 时，越靠近边界，hinge 越大。阈值 1 是 surrogate 的 margin 尺度，不应被理解为固定的像素安全距离。它把硬性的“是否被攻破”改成连续惩罚，让尚未错误分类、但已经缺少足够 margin 的样本也能获得优化信号。令

<div class="math-block">
$$
v_i(\theta)=\left[1-\underline m(x_i;\theta)\right]_+.
$$
</div>

因为 $[1-z]_+$ 随 $z$ 单调递减，有

<div class="math-block">
$$
\arg\max_i v_i(\theta)=
\arg\min_i\underline m(x_i;\theta),
\qquad
\max_i v_i(\theta)=
\left[1-\min_i\underline m(x_i;\theta)\right]_+.
$$
</div>

论文的理论依据就在这里：**降低最大 per-sample robust hinge，与提高数据集中最小 robust margin，是同一个尾部优化方向。** 这并不意味着论文已经证明 C²R 一定提高测试集最小 margin；它说明的是，如果要针对最坏样本设计代理目标，低 margin 样本应该得到更高优先级。

注意，论文先写出 robust hinge 的平均风险代理，再利用最大 per-sample hinge 与最小 margin 的单调对应关系，论证最差样本为何值得关注。这是一个关于排序和 tail objective 的理论动机，不等于 AAC 在有限训练中精确最小化这个最大 hinge。AAC 使用攻击器估计 margin，后续梯度也通过完整训练目标和合成集合间接作用，因此真正是否改善测试分布，仍必须看实验。

### 2.2 为什么简单平均可能漏掉危险样本

假设一个 batch 中有 99 个 margin 很大的样本，只有 1 个样本贴着决策边界。平均对齐目标可能主要被前 99 个样本的稳定信号决定，那个真正影响 robust error 的样本却没有得到足够更新。

平均损失并不是错误的目标：它有助于稳定训练、保持总体语义并降低方差。论文反对的是把平均对齐作为唯一鲁棒性目标，而不给低 margin 尾部额外优先级。C²R 保留 performance loss，再用 AAC 改变鲁棒更新顺序。

可以把这个差异想成“均值”和“最弱环节”的差异。平均量适合回答整体表现如何，却不一定能发现少数已经越界或即将越界的点。另一方面，如果只盯住单个最差点，训练又可能被噪声或异常样本牵着走。C²R 并没有删掉平均性能项，而是利用课程排序把难例放在当前训练过程前部，再用 CRL 在这些 batch 上补足局部类间对比。它是偏向尾部的优化策略，不是把整套学习目标改成只追逐一个样本。

需要区分三种说法：robust hinge 与最小 margin 的关系是理论推导；“低 margin 样本更值得先修”是对推导的直观解释；“这样做能提高测试集鲁棒准确率”则是需要实验支持的经验结论。

## 3. C²R 方法：困难样本课程与边界级对比学习

下面先给出方法总览，再分别解释 AAC、LS-PGD 和 CRL。图中的关键顺序是：先生成并排序对抗样本，再在排序后的 batch 上进行性能优化和对比鲁棒优化。具体来说，AAC 不增加一项独立 loss，而是重新安排训练时看到不同 adversarial companions 的先后；CRL 才是直接改变 embedding 的鲁棒目标。LS-PGD 位于攻击生成这一步，为 AAC 提供 margin proxy，同时控制每个 epoch 的计算成本。

<figure class="paper-figure paper-figure-wide">
  <img src="/posts/robust-distilled-datasets/assets/figures/c2r-overview.png" alt="C²R 方法总览：LS-PGD 生成对抗样本并计算分数，AAC 排序样本，CRL 对齐同类并分离异类">
  <figcaption><strong>图 1：C²R 的整体流程。</strong> 左侧的 LS-PGD 生成 adversarial companions 并计算 perturbation score，AAC 按 score 把低 margin 样本排在前面；右侧的 CRL 同时拉近 clean/adversarial 正对和同类样本，并推开异类竞争者。读者应关注的是“排序”和“对比”两条路径如何汇合到同一个蒸馏更新中。</figcaption>
</figure>

### 3.1 AAC：把困难样本放到训练前面

论文用 PGD 近似 robust margin 的内层最小化。给定步长 $\alpha$、迭代次数 $T$ 和投影算子 $\Pi_\Delta$，PGD 更新为

<div class="math-block">
$$
\delta_{t+1}=\Pi_\Delta\left(
\delta_t+\alpha\operatorname{sign}
\left(\nabla_x\ell(f_\theta(x+\delta_t),y)\right)
\right).
$$
</div>

拿到终点后，论文使用

<div class="math-block">
$$
\widehat m_{\mathrm{rob}}(x;\theta)=
 g_\theta(x+\delta_T)
$$
</div>

作为 robust margin 的估计，再定义 perturbation score：

<div class="math-block">
$$
 s(x)=\left[1-\widehat m_{\mathrm{rob}}(x;\theta)\right]_+.
$$
</div>

分数越大，代表攻击终点让样本越接近或越穿过边界。每个 epoch 中，论文先计算 score，再按分数从高到低排序 adversarial samples，按照这个顺序组成 mini-batch。它不是从易到难逐步提高攻击强度，而是根据当前模型对样本的估计 margin，优先处理最危险的样本。

为什么“排序”有机会帮助合成图像更新？DD 的更新不是独立拟合每张训练图，而是通过一批训练数据的梯度改变合成集合。先让低 margin 对抗样本进入这些更新，相当于优先把边界附近的梯度信息传给合成集合；后续 batch 再在已经发生变化的模型和合成图上继续优化。若把顺序打乱，训练预算仍然存在，但最危险样本不再获得这种先行影响。

AAC 的实现还使用 continuation 式 warm start：前一轮已经找到的扰动作为下一轮初始化，避免每次从头搜索；当缓存扰动已经足够强时，通过 adaptive early stop 直接复用，不再为该样本执行多余攻击步。这一点解释了 AAC 与 LS-PGD 的分工：AAC 决定课程顺序，LS-PGD 和 early stop 则减少更新这个顺序所需的攻击成本。论文没有把这种顺序解释为严格增加“样本权重”，它改变的是 mini-batch 的排列和后续更新轨迹。

这里的关键字是“估计”。PGD 终点不等于扰动集合中的全局最小 margin；非凸网络下可能存在更强但没有被找到的攻击。因此 AAC 提供的是攻击相关的排序信号，而不是精确测量或 certified bound。

### 3.2 LS-PGD：让每轮排序的成本可接受

完整多步 PGD 对每个样本都需要多次反向传播，而 DD 本身已经反复更新合成数据和网络。LS-PGD 缓存上一轮的扰动 $\hat\delta(x)$ 作为 warm start：如果当前损失仍然没有下降，就复用它；否则只做一次反向传播得到上升方向

<div class="math-block">
$$
v=\operatorname{sign}\left(
\nabla_x\ell(f_\theta(x+\hat\delta(x)),y)
\right).
$$
</div>

再沿 $v$ 尝试少量几何步长：

<div class="math-block">
$$
\mathcal S=\{\alpha\beta^q\mid q=0,\ldots,Z-1\},
\qquad \beta>1,
\qquad Z\in\{2,3\}.
$$
</div>

通过 forward-only 评估选择损失最大的投影候选：

<div class="math-block">
$$
\delta'=\arg\max_{\eta\in\mathcal S}
\ell\left(f_\theta\left(x+\Pi_\Delta(\hat\delta(x)+\eta v)\right),y\right).
$$
</div>

候选集合包含标准步长 $\alpha$，所以在相同 warm start 下不弱于一次普通 PGD step；但这不等于 LS-PGD 等价于多步 PGD，更不等于找到了全局最坏扰动。它的工程目标是用较少反向传播，得到足以进行排序的攻击。

这个成本控制在蒸馏场景尤其重要：外层要更新合成图像，训练中还要反复调用模型和攻击器；如果每次排序都对每个样本做完整多步 PGD，内层攻击可能占去很大比例的计算预算。LS-PGD 把“是否值得再做一次梯度更新”变成条件判断，把步长选择变成少量 forward-only probes。论文报告的约 `1.4x` speedup 是其具体设置下的效率结果，不应该解读成任何实现都能获得同样加速。

### 3.3 CRL：从类均值对齐转向实例级边界比较

已有鲁棒 DD 常用类均值对齐：设 $e(\cdot)$ 是 embedding，$X_c$ 和 $\widetilde X_c$ 是第 $c$ 类 clean/adversarial samples，则

<div class="math-block">
$$
\mathcal L_{\mathrm{rob}}=
\sum_{c=1}^{C}
\left\|
\mathbb E_{x\in X_c}[e(x)]-
\mathbb E_{\tilde x\in\widetilde X_c}[e(\tilde x)]
\right\|_2^2.
$$
</div>

这个目标可以让同一类别的两个均值接近，但会抹平类内子模式，也没有直接告诉模型哪个异类样本是当前实例的最强竞争者。C²R 将它替换为 supervised contrastive objective。

举一个简单例子：某类里有猫脸和猫侧身两种外观。如果 clean 样本主要是猫脸、对抗样本主要变成猫侧身，只要两组 embedding 的均值接近，均值损失仍可能很小；但具体样本之间的对应关系可能已经很差。对比损失把实例对和同类关系明确写进目标，并把不同类作为竞争对象，因而保留了更多样本级结构。它也有代价：如果 batch 中同类样本太少，正样本关系不足；如果异类候选不代表真正的近邻，边界排斥信号也会变弱。

对 batch $B=\{(x_i,y_i)\}_{i=1}^{M}$，每个 $x_i$ 有 adversarial companion $\tilde x_i$。正样本集合为

<div class="math-block">
$$
P(i)=\{\tilde x_i\}\cup
\{x_j,\tilde x_j\mid y_j=y_i,\ j\ne i\}.
$$
</div>

$\tilde x_i$ 要求同一个实例在扰动下保持稳定；同类 clean/adversarial 样本保留类内结构；异类样本及其 adversary 则作为负样本。令 $e(\cdot)$ 为 embedding，$\operatorname{sim}$ 为余弦相似度，$g_{i,a}=\operatorname{sim}(e(x_i),e(a))$，温度为 $\tau>0$，CRL 为

<div class="math-block">
$$
\mathcal L_{\mathrm{CRL}}=
\frac1M\sum_{i=1}^{M}
\left[-\sum_{a\in P(i)}\frac1{|P(i)|}
\log\frac{\exp(g_{i,a}/\tau)}
{\sum_{b\in A(i)}\exp(g_{i,b}/\tau)}\right].
$$
</div>

其中 $A(i)$ 是正样本与异类候选的并集。这个目标同时做两件事：提高 clean anchor 与 adversarial companion 的相似度，降低它与竞争类别的相似度。由于 softmax 分母会更强烈地惩罚高相似度负样本，最接近的 hard negative 得到更大的边界排斥压力。

从公式看，分子里的 $\exp(g_{i,a}/\tau)$ 是对一个正样本的相似度奖励：提高 anchor 与该正样本的相似度，会降低对应的负对数概率。分母则把所有候选放在同一个归一化竞争中；高相似度异类会占据更大概率质量，因此只把正对拉近而不管负样本是不够的。温度 $\tau$ 控制相似度差异在 softmax 中有多尖锐：较低温度会让高相似度候选更突出，也让 hard negative 更有影响；较高温度会平缓这种差异。它因此既是数值尺度参数，也是正负样本竞争强度的调节项。

CRL 是 embedding-space 目标，不等同于直接对每个 logit margin 加一个严格下界。论文的论证是：实例级对比使同类 clean/adversarial 表示稳定，并显式压低与竞争类相似度，因而与 robust margin 的竞争项相呼应。最终 margin 是否提升仍取决于表示如何被分类头使用，并由最终攻击评测验证。

CRL 的总目标仍保留普通性能项：

<div class="math-block">
$$
\mathcal L_{\mathrm{C^2R}}=
(1-\eta)\mathcal L_{\mathrm{perf}}
+\eta\mathcal L_{\mathrm{CRL}},
\qquad \eta\in[0,1].
$$
</div>

为扩大负样本范围，论文维护 class-balanced FIFO memory queue，用低维随机投影检索 top-$k$ hard negatives，最后仍用完整 embedding 计算损失。全量 batch 对比约为 $\mathcal O(M^2)$，检索后约为 $\mathcal O(Mk)$。代价是引入 queue 陈旧表示、投影误差和 top-$k$ 截断，因此它是效率近似，不是免费优化。

队列按类别平衡，避免少数大类在候选池里占据绝大多数位置。随机投影只用于快速筛候选，最终 CRL 仍在完整 embedding 上计算；这一点很重要，它表示近似发生在“找到哪些负样本”这一步，而不是把损失本身改成低维代理。论文还缓存 embedding 并以移动平均方式更新，从而在多个训练步之间摊薄 encoder forward 成本，同时当前 batch 的 embedding 仍参与梯度计算。

这一设计针对一个具体瓶颈：batch 内异类数量有限，最相似的竞争类别可能刚好不在当前 batch。跨 batch queue 扩大了候选范围，top-$k$ 则避免把所有历史 embedding 都放进每个 anchor 的分母。它依赖的假设是：低维代理相似度能够把真正有用的 hard negatives 召回出来，queue 中较旧的表示仍有参考价值。消融结果显示完整 hard-negative mining 在论文设置下优于只看 batch 的对比，但这些近似在不同数据规模和表示漂移速度下仍值得单独验证。

### 3.4 一次 C²R 训练迭代

把方法落成工程流程，可以概括为：先用 LS-PGD 生成 adversarial companions，计算 $s(x)$ 并按分数降序排列；再对当前 batch 构造自己的 adversary、同类正样本和 hard negatives；随后用 $\mathcal L_{\mathrm{C^2R}}$ 更新网络与合成图像，最后把当前 embedding 放入对应类别的 queue。

这里的变量之间有一条容易忽略的依赖链：当前网络决定攻击结果和 margin 排序；排序后的 adversaries 决定本轮的 batch 顺序；当前 batch 与 queue 决定 CRL 的正负关系；组合损失再同时影响网络和合成图像，下一轮的攻击和排序因此会变化。换言之，AAC 不是预先对真实数据做一次静态排序，而是随着训练状态更新的课程。

AAC 决定“哪些样本先被处理”，CRL 决定“这些样本要形成什么几何关系”。AAC 单独使用只是改变更新顺序，CRL 单独使用仍可能把预算分散给容易样本；论文认为两者结合才同时覆盖了 margin 尾部和决策边界。

## 4. 实验：提升发生在什么协议下

### 4.1 数据、架构、攻击与指标

主文覆盖 CIFAR-10、CIFAR-100、Tiny-ImageNet，以及六个 ImageNet-1K 十分类子集：ImageNette、ImageWoof、ImageFruit、ImageMeow、ImageSquawk 和 ImageYellow。ImageNet 子集图像统一调整到 $128\times128$；CIFAR 使用标准三块 ConvNet，高分辨率数据使用 ConvNet-D4/D5。

主实验的 IPC（images per class）为 CIFAR/Tiny-ImageNet 的 `1, 5, 10, 30, 50`，ImageNet 子集的 `1, 10, 50`。基线包括 MTT、SRe²L、D⁴M 和 ROME，它们分别代表训练轨迹匹配、squeeze-recover-relabel、扩散模型生成和鲁棒特征对齐等不同路线。

这些设置决定了结果该如何比较。IPC 控制每类有多少张合成图，backbone 决定模型容量和特征几何，攻击预算决定“被认为脆弱”的扰动范围；其中任一项不一致，准确率差异就不能简单归因于算法。论文选择了提供官方实现且能在兼容 benchmark 设置下比较的鲁棒 DD 基线，因此主表更适合回答“在相同协议下哪种方法更有效”，不适合回答“哪种方法在所有蒸馏范式下都最好”。

主文使用 FGSM、PGD、CW、VMIFGSM、Jitter 和 AutoAttack 等攻击。主表中，FGSM、PGD、CW、VMIFGSM 和 Jitter 使用 $\ell_\infty$ 扰动预算 $\varepsilon=2/255$；AutoAttack 的具体配置另见附录。主要结果平均五次独立运行。

论文同时报告 clean accuracy、robust accuracy 和 drop rate。后者定义为

<div class="math-block">
$$
\mathrm{DR}=\frac{\mathrm{Acc}_{\mathrm{clean}}-
\mathrm{Acc}_{\mathrm{robust}}}
{\mathrm{Acc}_{\mathrm{clean}}}\times100\%.
$$
</div>

robust accuracy 高表示攻击后仍能正确分类，DR 低表示从 clean 到 robust 的跌落较小；二者不能混用。一个 clean accuracy 很低的模型可能有较低 DR，但这并不代表它是好的蒸馏模型；因此要把绝对准确率和相对跌落放在一起看。

### 4.2 主结果与 drop rate

论文摘要和主文报告，在 CIFAR-10、CIFAR-100、Tiny-ImageNet 的主表对照中，C²R 相比已有鲁棒 DD 平均提升约 **2.8 个百分点**。这个数字依赖论文指定的数据集、IPC、网络和攻击设置，并不是跨任务常数。

下面选取 IPC=10 的几组结果，单位是测试准确率百分比，越高越好：

| 数据集 | IPC | 攻击 | ROME | C²R |
| --- | ---: | --- | ---: | ---: |
| CIFAR-10 | 10 | PGD | 24.01 | 29.12 |
| CIFAR-100 | 10 | PGD | 8.42 | 11.38 |
| Tiny-ImageNet | 10 | PGD | 1.36 | 3.83 |
| CIFAR-10 | 10 | FGSM | 25.72 | 28.24 |
| CIFAR-100 | 10 | CW | 9.37 | 12.72 |

CIFAR-10、IPC=10 的 clean accuracy 中，ROME 为 `47.94%`，C²R 为 `48.31%`。在这个设置下，鲁棒提升并不是简单牺牲 clean accuracy 换来的；但不能据此声称所有 IPC 和数据集都不存在权衡。

读表时应同时关注三件事。第一，C²R 的提升是否同时出现在 clean 与 robust 指标，而不是只靠降低 clean accuracy 换取较小 drop。第二，提升是否跨 FGSM、PGD、CW、VMI 和 Jitter 等攻击出现，还是只在一个训练相关攻击上出现。第三，提升是否在不同 IPC 和数据集上都保持，还是只由某个容易的子集贡献平均值。论文的主结果支持前两点在其协议下成立，但“平均 2.8 个百分点”不能替代逐数据集、逐攻击的检查。

论文还观察到 C²R 在多种攻击下都有收益，而不是只对训练时的某一个方向有效。这支持“蒸馏表示的局部结构得到改善”的解释，但这些攻击仍属于论文定义的图像威胁模型，不能外推为未知攻击下的普适保证。

<figure class="paper-figure paper-figure-wide">
  <img src="/posts/robust-distilled-datasets/assets/figures/drop-rate-cifar10.png" alt="CIFAR-10 上不同 IPC 的 PGD drop rate 曲线">
  <figcaption><strong>图 2：CIFAR-10 上的 PGD drop rate。</strong> 论文比较不同方法在 IPC 变化时 clean accuracy 到 robust accuracy 的相对跌落。应观察的是曲线的相对高度和随 IPC 的变化趋势，而不是把某个点读成所有数据集的固定安全阈值。</figcaption>
</figure>

在 PGD 评测中，论文报告 C²R 跨数据集的平均 DR 低于 `66.8%`，并且随着合成数据规模变化曲线更平。更准确的解读是：在相同评测协议下，C²R 的 clean-to-robust 性能下降更小、更稳定；不是说它在任何实际攻击中都只会损失不到某个固定比例的准确率。

图中的红色虚线代表完整数据训练模型的参考水平，曲线则展示不同蒸馏方法随 IPC 改变的相对跌落。这里有两个阅读重点：曲线越低，说明从 clean 到 robust 的损失相对越小；曲线越平，说明增加合成数据时鲁棒退化没有明显恶化。drop rate 仍然是归一化指标，必须与对应的 clean/robust accuracy 一起看。

### 4.3 合成图像与 IPC 的非单调性

论文在 IPC=50 下对比了 ROME 与 C²R 的合成图像。C²R 的样本通常呈现更清晰的对象结构、更连贯的颜色组成和更可识别的类别语义；ROME 的部分样本则出现颜色伪影或重复纹理。视觉质量只能作为定性证据，不能替代攻击评测，但它能帮助理解 AAC/CRL 可能如何改变蒸馏样本。

这些图不是自然图像质量排行榜，也不是对整个合成数据集的统计检验。它们展示的是论文挑选的数据集、方法和 IPC 下的一组可视化样本；读者可以用来观察对象结构、颜色组成和纹理重复，却不能仅凭几张图推出 robust accuracy。定量结果仍来自前面的攻击评测。

<div class="paper-figure-grid">
  <figure class="paper-figure">
    <img src="/posts/robust-distilled-datasets/assets/figures/rome-cifar100.png" alt="ROME 在 CIFAR-100 上生成的合成图像">
    <figcaption><strong>图 3a：ROME。</strong> 论文在 CIFAR-100、IPC=50 下的合成样本，可观察类内纹理重复和局部伪影。</figcaption>
  </figure>
  <figure class="paper-figure">
    <img src="/posts/robust-distilled-datasets/assets/figures/c2r-cifar100.png" alt="C²R 在 CIFAR-100 上生成的合成图像">
    <figcaption><strong>图 3b：C²R。</strong> 同一数据集和 IPC 下的样本，论文认为其对象结构与类别语义更清晰。</figcaption>
  </figure>
  <figure class="paper-figure">
    <img src="/posts/robust-distilled-datasets/assets/figures/c2r-imagenette.png" alt="C²R 在 ImageNette 上生成的合成图像">
    <figcaption><strong>图 3c：ImageNette。</strong> C²R 在更高分辨率子集上的定性结果，显示方法并不只在 CIFAR 图像上呈现可读结构。</figcaption>
  </figure>
</div>

IPC 从 1 增加到 10 或 50 并不保证 robust accuracy 单调增加。更多样本确实提供了更多训练信号，但也可能让不同类别的局部区域更密集、边界邻域重叠更多，增加攻击转移的机会。论文主文将 IPC 从 10 到 50 时的轻微下降解释为局部区域在边界附近变得更密集，邻近样本之间更容易发生对抗迁移。

这件事也说明 IPC 不是一个单调的“越大越好”旋钮。增加预算可以提供更丰富的类内变化，但如果新样本是冗余的，或者让合成分布在边界两侧互相挤压，clean/robust trade-off 可能变化。C²R 的 AAC 会优先处理低 margin 尾部，CRL 会尝试分离 hard negatives，但二者没有消除样本多样性和边界位置本身带来的限制。

## 5. 消融、效率与结论边界

论文在 IPC=10 下对 AAC 和 CRL 做组合消融，报告平均所有攻击的鲁棒准确率：

| AAC | CRL | CIFAR-10 | CIFAR-100 | Tiny-ImageNet | ImageNette |
| --- | --- | ---: | ---: | ---: | ---: |
| 否 | 否 | 24.81 | 9.40 | 1.43 | 23.76 |
| 是 | 否 | 26.73 | 11.45 | 2.73 | 25.27 |
| 否 | 是 | 26.04 | 10.98 | 2.10 | 24.65 |
| 是 | 是 | 28.41 | 12.30 | 3.74 | 26.62 |

AAC 单独有效，说明低 margin 样本的优先级有价值；CRL 单独有效，说明实例级对比和异类排斥比简单均值对齐更有用；两者联合最好，支持“样本选择”和“边界几何”是互补问题。

这张表还可以帮助排除一种误读：AAC 和 CRL 并不是两个可以互相替代的实现细节。AAC 主要改变输入给蒸馏更新的样本顺序，它回答“有限的优化预算首先花在哪里”；CRL 主要改变表示空间中的相对关系，它回答“这些样本之间应该怎样靠近或分离”。前者没有显式的负样本项，后者也不会自动知道哪些样本是当前最危险的。联合结果好于任一单独组件，正是因为它们作用在训练链条的不同位置。

论文还单独检查了 CRL 的 hard-negative mining。只使用当前 batch 的负样本时，CIFAR-10 的平均鲁棒准确率为 `28.15%`；使用 class-balanced queue 并检索 top-$k$ hard negatives 的完整 CRL 达到 `28.63%`，CIFAR-100、Tiny-ImageNet 和 ImageNette 也分别从 `11.82%/3.41%/25.96%` 提升到 `12.30%/3.74%/26.62%`。这说明 memory queue 不是纯粹的速度优化：跨 batch 找到更接近的异类，确实会改变边界附近的训练信号。不过，这个结果仍然只说明论文设置下的候选检索有效，不能保证任意随机投影或 queue 长度都能找到真正的最强竞争者。

LS-PGD 与标准多步 PGD 的比较也体现了效率折中：IPC=10 时，CIFAR-10 的平均鲁棒准确率为 `28.63/28.41`，CIFAR-100 为 `12.54/12.30`，Tiny-ImageNet 为 `3.92/3.74`，ImageNette 为 `26.82/26.62`，前者为标准多步 PGD，后者为 LS-PGD 结果。论文报告约 `1.4x` speedup。它支持“LS-PGD 足以提供排序信号”的判断，不支持“LS-PGD 可以替代最终评测中的强攻击”。

论文还从训练时间和鲁棒准确率的关系观察整体效率：AAC 通过 continuation initialization 和 adaptive early stop 减少重复的攻击更新，LS-PGD 把更多搜索变成 forward-only 的候选评估；CRL 则通过 top-$k$ 检索避免对所有历史负样本做两两比较。这里的效率收益不是来自删除鲁棒目标，而是来自减少冗余的攻击反向传播和负样本计算。实际部署时仍应分别测量攻击生成、embedding 检索和外层蒸馏更新的时间，不能只用一次总训练时间推断瓶颈。

鲁棒项权重也和蒸馏预算相关：IPC=1 和 10 时，$\eta=0.4$ 附近表现较好；IPC=50 时，$\eta=0.6$ 更有利。论文还指出这是一个相对平缓的最优区间，而非一个极其尖锐的单点。低 IPC 下，对比关系稀疏，过大的 CRL 可能对少量 pair 过拟合；IPC 增大后，更多同类和异类关系让更强的对比项更容易发挥作用。

这个超参数现象可以从两个方向理解。$\eta$ 太小，CRL 只提供很弱的局部几何约束，方法接近普通 performance alignment；$\eta$ 太大，训练会过度强调 clean/adversarial 表示不变性，可能压缩本应保留的类内变化，甚至损害 clean 分类。随着 IPC 增加，合成集合提供更多可比较的实例和 hard negatives，CRL 的信号更稳定，所以较大的权重更容易带来收益。工程上不应把 IPC=10 的 $\eta$ 当成所有预算的固定默认值。

**理论、算法、实验必须分开读。** 论文理论证明的是：在给定 robust margin 和 robust hinge 定义下，最大 per-sample hinge 对应最小 robust margin。AAC 用 PGD/LS-PGD 端点估计这个量并排序，是算法近似。C²R 在多个图像分类 benchmark 下取得更高 robust accuracy，是实验观察。

这三层之间存在一条没有被论文完全封闭的推理链：训练时只能看到有限的合成样本，攻击器只能近似求内层最小值，CRL 又是在 embedding 空间而不是直接在输入空间证明 margin。因此，“理论上应该优先低 margin 样本”不能直接等价为“训练后测试集的每个最小 margin 都提高了”。论文用跨攻击、跨 IPC 和消融实验来支持这种设计的有效性，但仍然是经验支持，不是端到端定理。

因此，论文没有证明：

- PGD 终点就是每个样本的真实最小 robust margin；
- AAC 一定最大化测试分布的最小 margin；
- CRL 在任意网络和数据分布上都扩大真实决策边界；
- C²R 对任意攻击都优于所有基线；
- 蒸馏集合因此获得 certified robustness。

还要注意攻击协议本身的边界。论文的主结果围绕图像分类、指定 ConvNet、有限 IPC 和明确的 $\ell_\infty$ threat model；AutoAttack、$4/255$ 和 $8/255$ 的附加结果扩大了覆盖范围，但仍不是对任意范数、任意语义扰动或物理攻击的保证。训练时的 LS-PGD 也只承担 margin 排序，不应拿它替代最终的强攻击评测。

作者还明确指出，当前方法主要验证中等分辨率图像分类和标准架构，尚未系统覆盖更高分辨率、检测、分割、图任务、持续学习、semantic/patch-based/distribution-shift 攻击或 certified robustness。附录提供了 $4/255$、$8/255$ 和 AutoAttack 的扩展结果，说明方法在更多设置下仍有优势，但没有消除这些适用边界。

## 6. 实践判断与总结

如果任务需要把训练集压缩到很小，并且下游模型面对明确的 $\ell_\infty$ 扰动，C²R 最值得借鉴的不是某个固定超参数，而是三条原则。

**先验证困难样本是否真的值得优化。** 记录攻击后的 margin 或 loss 分位数，检查低 margin 样本是否集中于真实困难模式，还是主要来自错误标签、异常图像和预处理问题。AAC 适合处理真实但易受攻击的样本，不负责数据清洗。实践中可以先画出 margin 的分位数曲线，比较课程排序前后的 tail 是否移动，而不是只看一个平均 robust accuracy。

**训练攻击和评测攻击分离。** 可以用 LS-PGD 生成排序信号，但最终评价应使用独立配置、更充分的 PGD、AutoAttack 或其他适合任务的强攻击。训练时最难的样本不一定是评测时最容易失败的样本。尤其要记录训练预算与评测预算、初始化方式、步数、是否 early stop，以及最终使用的 backbone，否则不同方法之间的数字很难公平比较。

**同时报告 clean accuracy、robust accuracy 和 drop rate。** 只报告攻击后准确率，可能看不出 clean performance 的牺牲；只报告 drop rate，又可能奖励一个原本就很差的模型。复现实验至少应保留 IPC、backbone、训练和评测攻击预算、攻击步数与随机种子，并同时报告不同 IPC 的曲线，而不是只挑一个最好的点。

如果换到语义扰动、patch attack、分布偏移或非视觉任务，还需要重新定义“正样本”和“最强竞争者”。C²R 的启发可以保留，但不能默认 logit margin、embedding cosine 和 $\ell_\infty$ 球仍然具有相同含义。一个合理的迁移方案应先验证攻击集合、margin proxy 和 hard-negative 检索是否仍与任务失败模式一致，再决定是否使用 AAC 或 CRL。

这篇论文真正改变的不是“给 DD 加一个 adversarial loss”，而是重新安排了鲁棒性优化的关注对象：理论上关注最小 robust margin，算法上优先处理低 margin adversaries，表示上同时保持 clean/adversarial 不变性并排斥 hard negatives。

如果把全文压缩成一句话，可以这样理解：**先用攻击器找出当前最容易越界的合成/对抗样本，再让这些样本在表示空间中靠近自己的扰动版本、远离最相似的异类，同时保留原本的分类性能目标。** AAC 解决“训练预算投向哪里”，CRL 解决“边界关系如何塑造”，LS-PGD 解决“这件事能否在可接受成本内反复做”。

它在论文协议下提供了较好的实验证据，但仍是一种面向图像分类的优化框架，不是通用安全保证。真正复现这篇工作时，最值得检查的也不是某个漂亮的单点数字，而是 margin 排序是否可靠、hard negatives 是否真的来自竞争类别、clean/robust trade-off 是否在不同 IPC 下稳定，以及独立强攻击能否复现论文观察到的优势。

<div class="citation-block">
<p><strong>论文</strong>：Muquan Li et al., <em>Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?</em>, ICML 2026 oral, arXiv:2605.20606v2。</p>
<p><strong>公开页面</strong>：<a href="https://icml.cc/virtual/2026/oral/71143">ICML 2026 页面</a> · <a href="https://arxiv.org/abs/2605.20606v2">arXiv v2</a> · <a href="https://github.com/SLGSP/CCR">官方代码仓库</a></p>
</div>
