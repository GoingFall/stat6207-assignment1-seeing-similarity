# 已执行实验协议 v3

当前路径：`data/protocol.json`、`data/protocol.sha256`、`data/test_manifest.json`。整理目录只改变文件位置，没有改写锁定 JSON、测试图像、来源标签或主结果。原 v2/v3 规划讨论保存在 `backup/v3-before-reorganization/Revised Experiment Protocol.md`。

## 授权与时序

用户在审核自动审计方案后要求“执行到完成并核验结果”。本设计是在看过 pilot 天花板效应后修订的，不能称为 pilot 前预注册。正式数据选择、图像与协议哈希先固定，之后运行三编码器推理。正式主分类固定 k=5，不通过验证集选择冠军，不用字母序指定最佳模型；网站初始设置为教学展示。

## 数据和自动审计

- PetImages v1：参考池 1000（每类500），测试500（每类250）；没有开发集或验证集。
- 候选为每类按固定种子抽取1800张，加全部 pilot PetImages。解码与EXIF校正RGB、文件/像素SHA-256、pHash Hamming<=10保守连通分组；每组最多保留固定排序的一个代表，排除连接pilot的组及确定的精确重复标签冲突组。
- ORB 为候选对补充证据（800特征、ratio 0.75、至少8匹配，RANSAC重投影阈值3）；不扩大pHash召回范围。审计3817张可解码候选、8对pHash候选、0解码失败，不代表全来源库无问题。
- 500张测试标签始终是 source_unverified。EfficientNet-B0独立审计汇总ImageNet猫281–285、家犬151–268概率；相反来源类别>=0.8且本类<=0.1标分歧。223一致、276不确定、1高置信分歧，全部保留。不使用模型输出改标/换图/剔除。
- Animals-10 v2原20张为定性OOD探针，不进入主测试、KNN投票参考库或显著性检验。

## 预处理和锁定条件

公共输入为EXIF校正RGB后直接双三次ImageOps.fit至224×224，再由各模型官方processor处理。这是执行时冻结的操作，不是历史提议的短边256再裁剪。

- 三个独立轨道：Gaussian blur radius 1/2；JPEG quality 50/20、4:2:0；灰127正方形side71/112、位置由42004+image ID的SHA派生。
- 每轨clean/mild/moderate，共享clean，6个非clean条件，不叠加；参考图保持clean。不声称人类仍能辨识被遮挡类别。
- 每编码器4500输入：1000参考、500clean测试、3000扰动测试。另20张OOD定性缓存。
- 推理前固定测试、manifest和variant hashes；评估入口核对文件字节和协议SHA。标签审计另存sidecar，不改清单。

## 抽样与评估

- ResNet-18 / DINOv2-S/14 / CLIP-B/32固定权重；L1/L2/cosine。ResNet-50未下载。
- seed1001–1010；每类10/25/50/100参考。同seed嵌套，无放回；不同seed允许重叠。距离并列稳定image ID排序；uniform KNN k=5。
- 主样本效率：clean各规模。主扰动：100/类、6条件。k敏感性只在clean/cosine/100每类，k={1,3,5,7,9}；unit ablation只在clean/100每类。
- accuracy主指标，macro-F1与P@5次指标；P@5比较须注意不同参考规模下覆盖比例改变。十seed标准差与query不确定性分别报告。
- 预定10图展示由seed42003选5猫5狗。检索及Grad-CAM展示使用其第一个ID；失败图按教学配置错误ID顺序，属于描述性检查。

## 推断范围

主检验端点为每query跨6非clean条件及10参考抽样的平均正确率。paired/source-class-stratified query bootstrap 5000次（42005）；所有模型、扰动随原query一起抽。双侧paired sign permutation 10000次（42006）、+1修正，需交换性假设。

三对cosine编码器比较与九对编码器内距离比较分别Holm校正。区间为边际95%区间，不是多重校正的同时区间。clean、单轨、样本量、P@5、k和归一化仅描述性。区间条件于固定参考池及十个已实现抽样，不能修复标签噪声、预训练重叠或合成扰动外推限制。

## 当前文件与验证

主程序依次见README：`prepare_data.py`、`evaluate.py`、`analyze.py`、`label_audit.py`；展示与报告为`export_site.py`、`visualize.py`、`explain.py`、`make_report.py`、`ood.py`。`verify.py`、`browser_check.py`、`delivery_check.py`、`review_report.py`负责数值、浏览器、图像重建与PDF渲染检查。报告按作业A1–A5、B1–B3、C编排；旧成本表及性能冠军结论只保留于backup历史。
