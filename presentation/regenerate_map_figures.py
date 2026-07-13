from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = Path(__file__).resolve().parent / "figures"

REGIONS_URL = (
    "https://raw.githubusercontent.com/openpolis/geojson-italy/master/"
    "geojson/limits_IT_regions.geojson"
)
PROVINCES_URL = (
    "https://raw.githubusercontent.com/openpolis/geojson-italy/master/"
    "geojson/limits_IT_provinces.geojson"
)

RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk_blue_white_red",
    [
        (0.000, "#08306b"),
        (0.180, "#08519c"),
        (0.340, "#2171b5"),
        (0.455, "#6baed6"),
        (0.492, "#d7ecff"),
        (0.500, "#ffffff"),
        (0.508, "#ffe0dc"),
        (0.545, "#fb6a4a"),
        (0.660, "#ef3b2c"),
        (0.820, "#cb181d"),
        (1.000, "#67000d"),
    ],
)


def load_geojson(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def normalize_name(value: object) -> str:
    text = str(value).strip().lower()
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2019", "'")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def polygon_rings(geometry: dict) -> list[list[tuple[float, float]]]:
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return []

    rings = []
    for polygon in polygons:
        if polygon:
            rings.append([(float(x), float(y)) for x, y in polygon[0]])
    return rings


def draw_map(
    *,
    features: list[dict],
    name_key: str,
    values: dict[str, float],
    output_path: Path,
    title: str,
    legend_title: str,
    vcenter: float,
    missing_color: str = "#e8e8e8",
    missing_edge_color: str = "#555555",
    missing_hatch: str | None = None,
) -> None:
    all_values = np.array(list(values.values()), dtype=float)
    norm = TwoSlopeNorm(
        vmin=float(all_values.min()),
        vcenter=float(vcenter),
        vmax=float(all_values.max()),
    )

    colored_patches = []
    patch_values = []
    missing_patches = []

    for feature in features:
        name = normalize_name(feature["properties"][name_key])
        value = values.get(name)
        target_patches = colored_patches if value is not None else missing_patches
        for ring in polygon_rings(feature["geometry"]):
            if len(ring) < 3:
                continue
            target_patches.append(Polygon(ring, closed=True))
            if value is not None:
                patch_values.append(value)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    if missing_patches:
        missing_collection = PatchCollection(
            missing_patches,
            facecolor=missing_color,
            edgecolor=missing_edge_color,
            linewidth=0.18,
            hatch=missing_hatch,
        )
        if missing_hatch:
            missing_collection.set_rasterized(True)
        ax.add_collection(missing_collection)

    colored_collection = PatchCollection(
        colored_patches,
        cmap=RISK_CMAP,
        norm=norm,
        edgecolor="#555555",
        linewidth=0.18,
    )
    colored_collection.set_array(np.array(patch_values, dtype=float))
    ax.add_collection(colored_collection)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=10)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=RISK_CMAP),
        ax=ax,
        fraction=0.045,
        pad=0.02,
        shrink=0.52,
    )
    tick_values = np.array(
        [
            float(all_values.min()),
            (float(all_values.min()) + vcenter) / 2,
            vcenter,
            (vcenter + float(all_values.max())) / 2,
            float(all_values.max()),
        ]
    )
    colorbar.set_ticks(tick_values)
    colorbar.ax.set_title(legend_title, fontsize=12, pad=8)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    region_data = pd.read_csv(OUTPUTS / "region_high_risk_cluster_share.csv")
    province_data = pd.read_csv(OUTPUTS / "province_risk.csv")

    region_values = {
        normalize_name(row.Region): float(row.high_risk_cluster_share)
        for row in region_data.itertuples(index=False)
    }
    province_values = {
        normalize_name(row.Province): float(row.distress_rate)
        for row in province_data.itertuples(index=False)
    }

    region_center = float(
        np.average(
            region_data["high_risk_cluster_share"],
            weights=region_data["n"],
        )
    )
    province_center = float(
        np.average(
            province_data["distress_rate"],
            weights=province_data["count"],
        )
    )

    regions = load_geojson(REGIONS_URL)
    provinces = load_geojson(PROVINCES_URL)

    draw_map(
        features=regions["features"],
        name_key="reg_name",
        values=region_values,
        output_path=FIGURES / "Regional risk.pdf",
        title="Regional high risk cluster share",
        legend_title="Risk",
        vcenter=region_center,
    )
    draw_map(
        features=provinces["features"],
        name_key="prov_name",
        values=province_values,
        output_path=FIGURES / "prov distress.pdf",
        title="Province distress rate",
        legend_title="Distress Rate",
        vcenter=province_center,
        missing_color="#f2f2f2",
        missing_edge_color="#4d4d4d",
        missing_hatch="////",
    )


if __name__ == "__main__":
    main()
