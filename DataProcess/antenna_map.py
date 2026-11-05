from uvfits_tools import UVFitsReader
import matplotlib.pyplot as plt
import numpy as np
def main():
    uvfits_path = (
        "/home/terakawayui/projects/kine3c345/observations/"
        "m87/uvfits/hops_3597_M87_LO.dcal_full.uvfits"
    )

    reader = UVFitsReader(uvfits_path)
    reader.summary()
    # reader.plot_uv_coverage(conjugate=False)

    dirty_image, dirty_beam = reader.make_dirty_image(
        pol="I",
        npix=128,
        fov_uas=200,
    )

    # 固定校准的 Stokes I -> Dirty image -> CLEAN；保留现有 CLEAN 参数。
    clean_image, residual, components = reader.clean(
        dirty_image,
        dirty_beam,
        gain=0.1,
        niter=1000,
        threshold=0.001,
    )
    # 调整：保存本次运行的恢复图、残差和分量，避免误读旧图片。
    from pathlib import Path
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, array, title in zip(axes, [clean_image, residual, components],
                                ["Restored CLEAN", "Residual", "Components"]):
        plot = ax.imshow(array, origin="lower", extent=[-100,100,-100,100], cmap="inferno")
        ax.set(title=title, xlabel="l (µas)", ylabel="m (µas)")
        fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("CleanImage.png"), dpi=200)
    plt.show()

    def print_stats(name, array):
        print(
            f"{name}: "
            f"shape={array.shape}, "
            f"min={np.nanmin(array):.6g}, "
            f"max={np.nanmax(array):.6g}, "
            f"max_abs={np.nanmax(np.abs(array)):.6g}, "
            f"center={array[array.shape[0] // 2, array.shape[1] // 2]:.6g}"
        )


    print_stats("dirty_image", dirty_image)
    print_stats("dirty_beam", dirty_beam)
    print("dirty_image finite:", np.isfinite(dirty_image).all())
    print("dirty_beam finite:", np.isfinite(dirty_beam).all())

    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # for ax, image, title in zip(
    #     axes,
    #     [clean_image, residual, components],
    #     ["CLEAN image", "Residual", "CLEAN components"],
    # ):
    #     plot = ax.imshow(
    #         image,
    #         origin="lower",
    #         extent=[-100, 100, -100, 100],
    #         cmap="inferno",
    #     )
    #     ax.set_title(title)
    #     ax.set_xlabel("l (µas)")
    #     ax.set_ylabel("m (µas)")
    #     fig.colorbar(plot, ax=ax)

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()