# CLEAN 调整说明

- CLEAN 分量使用峰值归一化高斯波束卷积后再加残差。默认采用 dirty beam 中心局部曲率估计椭圆高斯（不是完整主瓣拟合）；可通过 restoring_fwhm_pixels 指定圆高斯 FWHM（像素）。
- make_dirty_image 记录 uv 采样；同一 reader 随后 clean 时计算覆盖全部像素差的 PSF，修复远处旁瓣扣除不完整。外部传入的图像仍使用有限 PSF，且不应复用曾对其他数据成像的 reader。
- 增加 clean_mask、输入有限性与阈值检查。
- 入口保存 CleanImage.png 三联图；增加 scipy 显式依赖。

## 尚需确认

当前仍取固定通道和 pol=0，需先根据真实 FITS 的 STOKES/FREQ 轴确认含义。未自动进行自校准或改变原始权重。残差保持 dirty beam 单位，未做残差面积重标定，不适合直接据此精确测量积分流量。恢复波束和 CLEAN 参数不保证恢复 M87 环。

## 简化的 Stokes I 成像流程

默认入口现采用严格 I=(RR+LL)/2，仅保留两手均有效的记录。
独立噪声、逆方差权重假设下，wI=4/(1/wRR+1/wLL)。不使用单手近似 I。
假定复增益完全准确，不求解任何台站增益，不使用闭合量。
流程为自然权重 Dirty image -> Högbom CLEAN -> 高斯恢复加残差。
运行：MPLBACKEND=Agg uv run python DataProcess/antenna_map.py。
之前关于仍用 pol=0 的描述已由本节取代；诊断脚本不属于默认入口。
