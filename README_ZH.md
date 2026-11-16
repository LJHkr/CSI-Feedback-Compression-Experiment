# CSI反馈压缩实验

[English](README.md)

两个脚本：`csi_feedback.py` 做实验，`plot_results.py` 读CSV画图。参数都在脚本开头，可以直接改。

## 运行

```powershell
python -m pip install numpy pandas matplotlib
python experiment/csi_feedback.py
python experiment/plot_results.py
```

结果保存在 `experiment/results/`。数值实验使用NumPy，绘图使用Pandas和Matplotlib。

## 本轮参数

|参数|取值|
|---|---|
|信道维数|16个复数分量|
|训练／验证／测试样本|每个seed分别8000／2000／5000；每种测试相关性各5000|
|训练和验证相关性|rho=0.85|
|测试相关性|rho=0、0.5、0.85|
|上行反馈预算|32、64、96、128 bit|
|候选保留维数|2、4、6、8、12、16|
|每个实部／虚部的最高精度|16 bit|
|发射SNR|10 dB|
|波束码本|64个固定的单位范数过采样DFT波束|
|独立重复|10个seed，随机数种子2026到2035|

信道为零均值复高斯向量。协方差为 `R[i,j] = rho^abs(i-j) * exp(0.65j*(i-j))`，对角线为1，因此不同rho保持相同平均信道能量。0.65是固定的空间相位斜率。这些参数为本实验选用的示例设置。信道已知，反馈索引无误；本轮观察压缩和量化产生的误差。

## 三种方案

- `full_uniform`：全维信道的实部、虚部做均匀标量量化，所有分量等位。
- `pca_equal`：训练集拟合复数PCA，前k个分量尽量等位；不能整除时，前面的分量各多一位。
- `pca_unequal`：先估计各PCA分量不同精度下的训练量化MSE，每次把一对位分给当前误差下降最大的分量。一次增量同时给实部和虚部各加1 bit。

若每个复数分量的实部、虚部各使用b_i位，总载荷是 `B = 2*sum(b_i)`。零位分量重构为零，再恢复共同的训练均值。量化器使用等间隔格点；范围从训练标准差的1.5、2、2.5、3、3.5、4倍中选取，每档精度分别用训练MSE选择。

k=2在96和128 bit下不能满足每轴最多16 bit的设定，因此不参与这两档预算的选参。

主实验按验证集NMSE选择k。均值、PCA基、量化范围、位分配和k在测试前确定，三种测试rho使用同一套参数。各方法在同一seed下共用数据和波束码本。

补充对照固定64 bit，比较按验证集NMSE选k和按验证集波束速率损失选k，其他处理保持相同。

## 指标和共享成本

NMSE先对每个测试向量计算 `sum(abs(h-h_hat)^2)/sum(abs(h)^2)`，再对5000个样本平均。每次运行的平均NMSE转dB后，图中对10次运行的dB值取均值。

各方法先用重构信道选波束，再在真实信道上计算 `log2(1 + 10*abs(h^H*w)^2)`。速率损失的参考是同一码本中由真实CSI选择的波束。`beam_agreement`是与该参考选择一致的比例。

每种运行点使用固定反馈配置，因此每份报告只传上述B位索引。下行共享成本单独统计：均值、量化范围和PCA基按实际float64每实数64 bit计，PCA分配每项5 bit、分量数5 bit。全维方案使用已知坐标基，不传PCA矩阵。共享图将一个配置的成本除以10、100、1000、10000份报告，不与固定上行预算合并。

均值与区间以seed为单位计算。图中为10个运行值的均值和95% t区间（t_9=2.262），配对差异表先在同seed下相减再求区间。

## 结果文件

|文件|内容|
|---|---|
|`run_results.csv`|主实验360行，三方法、四预算、三种测试rho、10个seed|
|`dimension_results.csv`|64 bit下的维数扫描，120行|
|`objective_results.csv`|64 bit下两种选参目标的补充对照，120行|
|`allocations.csv`|各运行点选择的k、实际非零分量数、实部／虚部位数、共享成本|
|`eigenvalues.csv`|每次训练得到的PCA特征值|
|`summary.csv`、`paired_differences.csv`|主实验和维数扫描的汇总与配对区间|
|`objective_summary.csv`、`objective_paired.csv`|选参目标对照的汇总与配对区间|
|`budget.png/pdf`|匹配相关性下，预算与NMSE、速率损失|
|`dimensions.png/pdf`|维数与NMSE、速率损失|
|`mismatch.png/pdf`|训练／测试相关性不一致的影响|
|`selection_objective.png/pdf`|按NMSE或按速率损失选择维数|
|`allocation_and_sharing.png/pdf`|位分配示例及下行共享成本摊销|
