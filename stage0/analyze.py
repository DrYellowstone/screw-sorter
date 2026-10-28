"""阶段 0：把每张照片里的零件分割出来，算几何和颜色特征，看 4 类能不能分开。

用法：
    python stage0/analyze.py data/stage0 --out out/stage0

输入目录下每个子文件夹名就是类别（metric_screw / imperial_screw / metric_nut /
imperial_nut；mixed 等其他文件夹也会处理，标为未知类别）。
输出：
    features.csv      每个零件一行
    annotated/*.jpg   每张图标出检测到的零件和量出的长度
    scatter.png       颜色 vs 形状、长度直方图
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

COIN_MM = 25.0  # 1 元硬币直径
KNOWN = ["metric_screw", "imperial_screw", "metric_nut", "imperial_nut"]
MAX_SIDE = 2000  # 手机原图太大，先缩小


def load(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def segment(img):
    """背景是纯色纸：离背景颜色足够远的像素就是前景。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.median(lab.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    d8 = np.clip(dist * 3, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def blobs(mask, min_area):
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(contours):
        if hier[0][i][3] != -1:  # 是某个外轮廓里的孔，跳过
            continue
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hole = 0.0
        child = hier[0][i][2]
        while child != -1:
            hole += cv2.contourArea(contours[child])
            child = hier[0][child][0]
        out.append((c, area, hole))
    return out


def features(img, c, area, hole):
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    length, width = max(w, h), max(min(w, h), 1e-6)
    perim = cv2.arcLength(c, True)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(m, [c], -1, 255, -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2][m > 0]
    s = hsv[..., 1][m > 0]
    return {
        "cx": cx, "cy": cy,
        "length_px": length, "width_px": width,
        "elongation": length / width,
        "circularity": 4 * np.pi * area / max(perim, 1e-6) ** 2,
        "hole_ratio": hole / area,
        "v_mean": float(v.mean()), "v_p90": float(np.percentile(v, 90)),
        "s_mean": float(s.mean()),
        "area_px": area,
    }


def find_coin(feats):
    """硬币：最圆、没有孔的零件里面积最大的那个。"""
    cands = [i for i, f in enumerate(feats)
             if f["circularity"] > 0.8 and f["hole_ratio"] < 0.02 and f["elongation"] < 1.15]
    if not cands:
        return None
    return max(cands, key=lambda i: feats[i]["area_px"])


def shape_guess(f):
    if f["hole_ratio"] > 0.05 or f["elongation"] < 1.6:
        return "nut"
    return "screw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/stage0"))
    ap.add_argument("--min-area", type=float, default=150, help="小于这个像素面积的当噪点")
    args = ap.parse_args()

    (args.out / "annotated").mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic"}:
            continue
        img = load(path)
        if img is None:
            print(f"读不了 {path}（HEIC 要先在手机上改成“兼容性最佳”/JPG）")
            continue
        label = path.parent.name if path.parent.name in KNOWN else "unknown"
        found = blobs(segment(img), args.min_area)
        feats = [features(img, c, a, h) for c, a, h in found]
        coin = find_coin(feats)
        mm_per_px = COIN_MM / feats[coin]["length_px"] if coin is not None else None

        vis = img.copy()
        for i, ((c, _, _), f) in enumerate(zip(found, feats)):
            if i == coin:
                cv2.drawContours(vis, [c], -1, (255, 0, 255), 2)
                continue
            kind = shape_guess(f)
            f.update(file=str(path.relative_to(args.root)), label=label, shape_guess=kind,
                     length_mm=f["length_px"] * mm_per_px if mm_per_px else None)
            rows.append(f)
            color = (0, 200, 0) if kind == "screw" else (0, 140, 255)
            cv2.drawContours(vis, [c], -1, color, 2)
            txt = f"{f['length_mm']:.1f}mm" if mm_per_px else kind
            cv2.putText(vis, txt, (int(f["cx"]), int(f["cy"])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
        if mm_per_px is None:
            print(f"{path}: 没找到硬币，长度只有像素值")
        name = str(path.relative_to(args.root)).replace("/", "__").replace("\\", "__")
        cv2.imwrite(str(args.out / "annotated" / (Path(name).stem + ".jpg")), vis)
        print(f"{path}: {len(feats) - (coin is not None)} 个零件")

    if not rows:
        print("没检测到零件")
        return
    keys = ["file", "label", "shape_guess", "length_mm", "length_px", "width_px", "elongation",
            "circularity", "hole_ratio", "v_mean", "v_p90", "s_mean", "area_px", "cx", "cy"]
    with open(args.out / "features.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    plot(rows, args.out / "scatter.png")
    print(f"共 {len(rows)} 个零件，结果在 {args.out}")


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"metric_screw": "#222222", "imperial_screw": "#9aa4ad",
              "metric_nut": "#c0392b", "imperial_nut": "#2e86c1", "unknown": "#e0b84c"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for lab, col in colors.items():
        sel = [r for r in rows if r["label"] == lab]
        if not sel:
            continue
        ax[0].scatter([r["elongation"] for r in sel], [r["v_mean"] for r in sel],
                      c=col, label=lab, s=18, edgecolors="k", linewidths=0.3)
    ax[0].set_xlabel("elongation (length / width)")
    ax[0].set_ylabel("mean brightness V")
    ax[0].set_title("shape vs colour: screws right, nuts left; black low, silver high")
    ax[0].legend()

    screws = [r for r in rows if r["shape_guess"] == "screw" and r["length_mm"] is not None]
    for lab in ["metric_screw", "imperial_screw", "unknown"]:
        vals = [r["length_mm"] for r in screws if r["label"] == lab]
        if vals:
            ax[1].hist(vals, bins=40, alpha=0.6, color=colors[lab], label=lab)
    ax[1].set_xlabel("screw length (mm)")
    ax[1].set_title("length histogram: look for 3 clusters")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
