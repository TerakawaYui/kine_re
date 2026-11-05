from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


class UVFitsReader:
    """读取并检查 AIPS Random Groups 格式的 UVFITS 文件。"""

    def __init__(self, path):
        self.path = Path(path)

        with fits.open(self.path) as hdul:
            hdu = hdul[0]
            groups = hdu.data

            self.header = hdu.header.copy()

            # 每条记录的随机组参数
            self.u_sec = np.asarray(groups.par("UU---SIN")).copy()
            self.v_sec = np.asarray(groups.par("VV---SIN")).copy()
            self.w_sec = np.asarray(groups.par("WW---SIN")).copy()
            self.baseline = np.asarray(groups.par("BASELINE")).copy()

            # 数据最后一维的三个分量：实部、虚部、权重
            self.data = np.asarray(groups.data).copy()

        self.frequency_hz = float(self.header["CRVAL4"])

    @property
    def n_records(self):
        """UVFITS 中的记录数。"""
        return len(self.u_sec)

    @property
    def visibilities(self):
        """返回复可见度数组：实部 + 1j × 虚部。"""
        return self.data[..., 0] + 1j * self.data[..., 1]

    @property
    def weights(self):
        """返回可见度权重数组。"""
        return self.data[..., 2]

    @property
    def u_mlambda(self):
        """返回以百万波长（Mλ）表示的 u 坐标。"""
        return self.u_sec * self.frequency_hz / 1e6

    @property
    def v_mlambda(self):
        """返回以百万波长（Mλ）表示的 v 坐标。"""
        return self.v_sec * self.frequency_hz / 1e6

    def summary(self):
        """打印文件的基本信息。"""
        print(f"文件：{self.path}")
        print(f"目标：{self.header.get('OBJECT', '未知')}")
        print(f"记录数：{self.n_records}")
        print(f"参考频率：{self.frequency_hz / 1e9:.3f} GHz")
        print(f"DATA 数组形状：{self.data.shape}")
        print(f"复可见度数组形状：{self.visibilities.shape}")

    def plot_uv_coverage(self, conjugate=False, save_path=None):
        """绘制 UV 采样位置。

        参数
        ----
        conjugate : bool
            是否额外绘制 (-u, -v) 共轭对称位置。
        save_path : str 或 Path，可选
            如果提供路径，则将图保存到该位置。
        """
        u = self.u_mlambda
        v = self.v_mlambda

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(u, v, s=4, alpha=0.55, label="observed samples")

        if conjugate:
            ax.scatter(-u, -v, s=4, alpha=0.35, label="conjugate points")

        ax.set(
            xlabel="u (Mλ)",
            ylabel="v (Mλ)",
            title=f"UV coverage: {self.header.get('OBJECT', 'source')}",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True)
        ax.legend()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()
        return fig, ax

    def make_dirty_image(self, pol="I", npix=128, fov_uas=200):
        """计算 dirty image 和对应的 dirty beam，返回 (image, beam)。"""

        # UV 坐标：秒 × Hz = 波长
        u = self.u_sec * self.frequency_hz
        v = self.v_sec * self.frequency_hz

        # 从 DATA 中取出偏振轴和 [real, imag, weight] 轴
        data = self.data[:, 0, 0, 0, 0, :, :]

        # Stokes I 使用严格半和，只保留 RR、LL 同时有效的记录。
        # 假定增益已知且无误差；不执行自校准，也不使用闭合量。
        if pol == "I":
            codes = self.header["CRVAL3"] + (
                np.arange(data.shape[1]) + 1 - self.header.get("CRPIX3", 1)
            ) * self.header["CDELT3"]
            if -1 not in codes or -2 not in codes:
                raise ValueError("严格构造 I 需要 RR 和 LL")
            rr = data[:, np.flatnonzero(codes == -1)[0], :]
            ll = data[:, np.flatnonzero(codes == -2)[0], :]
            pair_valid = ((rr[:,2] > 0) & (ll[:,2] > 0)
                          & np.isfinite(rr).all(axis=1) & np.isfinite(ll).all(axis=1))
            vis = np.full(len(data), np.nan, dtype=complex)
            weight = np.zeros(len(data), dtype=float)
            vis[pair_valid] = ((rr[pair_valid,0] + 1j*rr[pair_valid,1])
                              + (ll[pair_valid,0] + 1j*ll[pair_valid,1])) / 2
            weight[pair_valid] = 4 / (1/rr[pair_valid,2] + 1/ll[pair_valid,2])
            print(f"Stokes I: {pair_valid.sum()}/{len(data)} paired RR/LL records")
        else:
            if not isinstance(pol, (int, np.integer)) or not 0 <= pol < data.shape[1]:
                raise ValueError('pol 必须为 "I" 或有效相关积索引')
            vis = data[:, pol, 0] + 1j * data[:, pol, 1]
            weight = data[:, pol, 2]

        valid = (
            (weight > 0)
            & np.isfinite(weight)
            & np.isfinite(vis.real)
            & np.isfinite(vis.imag)
            & np.isfinite(u)
            & np.isfinite(v)
        )
        u, v, vis, weight = u[valid], v[valid], vis[valid], weight[valid]

        if len(vis) == 0:
            raise ValueError("没有权重为正且数据有效的采样点")

        self._imaging_samples = (u.copy(), v.copy(), weight.copy(), npix, fov_uas)

        # 图像平面坐标，单位为弧度
        uas_to_rad = np.deg2rad(1 / 3600 / 1e6)
        fov_rad = fov_uas * uas_to_rad
        coords = (np.arange(npix) - npix // 2) * (fov_rad / npix)
        l, m = np.meshgrid(coords, coords)

        dirty_image = np.zeros((npix, npix), dtype=np.float64)
        dirty_beam = np.zeros((npix, npix), dtype=np.float64)

        chunk_size = 256

        for start in range(0, len(vis), chunk_size):
            stop = start + chunk_size

            phase = 2j * np.pi * (
                u[start:stop, None, None] * l[None, :, :]
                + v[start:stop, None, None] * m[None, :, :]
            )
            phasor = np.exp(phase)

            block_weights = weight[start:stop, None, None]

            # Dirty image: 逆傅里叶求和
            dirty_image += np.sum(
                block_weights
                * np.real(vis[start:stop, None, None] * phasor),
                axis=0,
            )

            # Dirty beam: 用单位可见度代替实际 vis
            dirty_beam += np.sum(
                block_weights * np.real(phasor),
                axis=0,
            )

        # 两张图使用相同的归一化
        normalization = np.sum(weight)
        dirty_image /= normalization
        dirty_beam /= normalization

        # 画图检查
        extent = [-fov_uas / 2, fov_uas / 2, -fov_uas / 2, fov_uas / 2]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        image_plot = axes[0].imshow(
            dirty_image,
            origin="lower",
            extent=extent,
            cmap="inferno",
        )
        axes[0].set_title(f"Dirty image, polarization index {pol}")
        axes[0].set_xlabel("l (µas)")
        axes[0].set_ylabel("m (µas)")
        fig.colorbar(image_plot, ax=axes[0], label="Visibility units")

        beam_plot = axes[1].imshow(
            dirty_beam,
            origin="lower",
            extent=extent,
            cmap="inferno",
        )
        axes[1].set_title("Dirty beam")
        axes[1].set_xlabel("l (µas)")
        axes[1].set_ylabel("m (µas)")
        fig.colorbar(beam_plot, ax=axes[1], label="Normalized response")

        plt.tight_layout()
        plt.show()

        return dirty_image, dirty_beam

    def clean(
    self,
    dirty_image,
    dirty_beam,
    gain=0.1,
    niter=1000,
    threshold=None,
    restoring_fwhm_pixels=None,
    clean_mask=None,
    ):
        """对 dirty image 做基础 Högbom CLEAN。

        返回：
            clean_image: CLEAN 分量恢复后与 residual 相加的图像
            residual:    迭代后剩余的残差图
            components:  CLEAN 分量图
        """
        residual = np.asarray(dirty_image, dtype=float).copy()
        beam = np.asarray(dirty_beam, dtype=float)

        if residual.shape != beam.shape:
            raise ValueError("dirty_image 和 dirty_beam 的形状必须相同")
        if not 0 < gain <= 1:
            raise ValueError("gain 必须在 (0, 1] 范围内")

        if residual.ndim != 2 or not np.isfinite(residual).all() or not np.isfinite(beam).all():
            raise ValueError("输入必须为有限的二维数组")
        if niter < 0 or (threshold is not None and threshold < 0):
            raise ValueError("niter 和 threshold 必须非负")
        mask = np.ones(residual.shape, dtype=bool) if clean_mask is None else np.asarray(clean_mask, dtype=bool)
        if mask.shape != residual.shape or not mask.any():
            raise ValueError("clean_mask 必须与图像同形且至少包含一个像素")

        ny, nx = residual.shape
        cy, cx = ny // 2, nx // 2
        components = np.zeros_like(residual)

        # 以 dirty beam 中心归一化
        peak_beam = beam[cy, cx]
        if not np.isfinite(peak_beam) or abs(peak_beam) < 1e-12:
            raise ValueError("dirty beam 中心值无效或接近零")
        beam = beam / peak_beam

        # 在中心附近取一个有限大小的 beam patch，
        # 平移时将它放到 residual 峰值位置。
        patch_radius_y = min(cy, ny - cy - 1)
        patch_radius_x = min(cx, nx - cx - 1)
        beam_patch = beam[
            cy - patch_radius_y : cy + patch_radius_y + 1,
            cx - patch_radius_x : cx + patch_radius_x + 1,
        ]

        # 调整：从实际 uv 采样生成完整像素差 PSF，避免边缘分量遗漏远处旁瓣。
        samples = getattr(self, "_imaging_samples", None)
        if samples is not None and samples[3] == nx == ny:
            u, v, weights, _, fov = samples
            step = np.deg2rad(fov / 3600 / 1e6) / nx
            yy, xx = np.mgrid[-(ny-1):ny, -(nx-1):nx]
            beam_patch = np.zeros(xx.shape, dtype=float)
            for start in range(0, len(u), 64):
                sl = slice(start, start+64)
                phase = 2*np.pi*step*(u[sl,None,None]*xx + v[sl,None,None]*yy)
                beam_patch += np.sum(weights[sl,None,None]*np.cos(phase), axis=0)
            beam_patch /= weights.sum()
            patch_radius_y, patch_radius_x = ny-1, nx-1

        if threshold is None:
            threshold = 0.0

        for iteration in range(niter):
            peak_index = np.unravel_index(np.argmax(np.where(mask, np.abs(residual), -np.inf)), residual.shape)
            py, px = peak_index
            peak_value = residual[py, px]

            if abs(peak_value) <= threshold:
                print(f"CLEAN 在第 {iteration} 次迭代达到阈值。")
                break

            component = gain * peak_value
            components[py, px] += component

            # 只在图像范围内减去平移后的 beam patch
            y0 = max(0, py - patch_radius_y)
            y1 = min(ny, py + patch_radius_y + 1)
            x0 = max(0, px - patch_radius_x)
            x1 = min(nx, px + patch_radius_x + 1)

            by0 = y0 - (py - patch_radius_y)
            by1 = by0 + (y1 - y0)
            bx0 = x0 - (px - patch_radius_x)
            bx1 = bx0 + (x1 - x0)

            residual[y0:y1, x0:x1] -= (
                component * beam_patch[by0:by1, bx0:bx1]
            )
        else:
            print(f"CLEAN 达到最大迭代次数：{niter}")

        # 调整：恢复 CLEAN 分量，再加回 dirty-beam 单位的残差。
        # 调整：用主瓣局部曲率估计椭圆高斯恢复波束；也可显式指定圆高斯 FWHM。
        if restoring_fwhm_pixels is None:
            if min(ny, nx) < 3 or beam[cy, cx] <= 0:
                raise ValueError("无法估计恢复波束，请指定 restoring_fwhm_pixels")
            hxx = beam[cy, cx+1] - 2 + beam[cy, cx-1]
            hyy = beam[cy+1, cx] - 2 + beam[cy-1, cx]
            hxy = (beam[cy+1,cx+1] - beam[cy+1,cx-1] - beam[cy-1,cx+1] + beam[cy-1,cx-1]) / 4
            precision = -np.array([[hxx, hxy], [hxy, hyy]])
            if np.min(np.linalg.eigvalsh(precision)) <= 0:
                raise ValueError("主瓣曲率无效，请指定 restoring_fwhm_pixels")
        else:
            if not np.isfinite(restoring_fwhm_pixels) or restoring_fwhm_pixels <= 0:
                raise ValueError("restoring_fwhm_pixels 必须为正数")
            sigma = restoring_fwhm_pixels / np.sqrt(8 * np.log(2))
            precision = np.eye(2) / sigma**2
        from scipy.signal import fftconvolve
        # 奇数尺寸核使偶数尺寸图像也保持像素中心对齐；峰值归一化保留 Jy/beam 约定。
        yy, xx = np.mgrid[-(ny-1):ny, -(nx-1):nx]
        restoring_beam = np.exp(-0.5 * (precision[0,0]*xx**2 + 2*precision[0,1]*xx*yy + precision[1,1]*yy**2))
        clean_image = fftconvolve(components, restoring_beam, mode="same") + residual

        return clean_image, residual, components