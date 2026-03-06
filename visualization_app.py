"""
ANSWER Lite — irodori 18バンド分光データ可視化（日英対応）
B2B SaaS スタイルのプロトタイプ（Streamlit）
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =============================================================================
# 定数
# =============================================================================
WAVELENGTHS = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]
BAND_COLS = [str(w) for w in WAVELENGTHS]
RED_BAND, NIR_BAND = "680", "810"

# =============================================================================
# 多言語（日英）
# =============================================================================
TRANSLATIONS = {
    "ja": {
        "app_subtitle": "irodori 18バンド分光データの可視化・簡易解析",
        "page_title": "ANSWER Lite — irodori可視化",
        "sidebar_dataset": "データセット",
        "upload_csv": "CSVをアップロード",
        "err_not_irodori": "このCSVはirodori形式ではありません。",
        "err_bands_missing": "波長列（410〜940nm）が不足しています。",
        "btn_load_sample": "サンプルデータを読み込む",
        "msg_loaded": "読み込みました",
        "or_load_sample": "または「サンプルデータを読み込む」",
        "sidebar_samples_filter": "サンプル・フィルタ",
        "hint_load_first": "データを読み込むと選択できます",
        "sidebar_sample_select": "サンプル選択",
        "include_samples": "表示に含めるサンプル",
        "sidebar_filter": "フィルタ",
        "filter_by_label": "ラベルで絞る",
        "filter_by_target": "ターゲットで絞る",
        "all": "（すべて）",
        "sidebar_pages": "ページ",
        "hint_tabs": "メインのタブで切り替え",
        "kpi_samples": "サンプル数",
        "kpi_labels": "ラベル数",
        "kpi_selected": "選択数",
        "kpi_status": "状態",
        "status_ready": "準備完了",
        "status_no_data": "データなし",
        "viz_params": "可視化パラメータ",
        "norm": "正規化",
        "norm_none": "なし",
        "norm_minmax": "Min-Max (0–1)",
        "norm_zscore": "Z-score",
        "smooth": "平滑化",
        "smooth_none": "なし",
        "smooth_3": "3点移動平均",
        "smooth_5": "5点移動平均",
        "display_unit": "表示単位",
        "unit_raw": "反射率（そのまま）",
        "unit_pct": "百分率 (%)",
        "viz_note_ndvi": "※NDVI・Power Dialは生データで計算します。",
        "op_help_title": "操作の説明",
        "op_help_norm": "正規化はサンプル間のスケール差を揃え、形状（スペクトルの傾向）を比較しやすくします。",
        "op_help_norm_minmax": "Min-Max: 各波長を0–1にスケーリングします（表示範囲を揃える用途）。",
        "op_help_norm_zscore": "Z-score: 各波長を平均との差/標準偏差で標準化します（平均との差の強弱を見やすくします）。",
        "op_help_smooth": "平滑化（移動平均）はノイズを減らし、スペクトルの大まかな形状を見やすくします。",
        "op_help_plot": "棒=離散的に強度を比較 / 線=連続スペクトルとして形状を比較できます。",
        "plot_type": "プロット種類",
        "plot_bar": "棒（Bar）",
        "plot_line": "線（Line）",
        "plot_line_markers": "線＋点（Line+Markers）",
        "overlay_title": "重ね描き（複数サンプル）",
        "overlay_hint": "複数サンプルを同一軸に重ねて、スペクトル形状の違いを直接比較します。",
        "max_overlay": "最大表示数",
        "diff_tab": "差分解析",
        "diff_title": "スペクトル差分解析",
        "diff_desc": "2グループ（A/B）の平均スペクトル差や、波長ごとの差の大きさ（効果量など）を算出します。",
        "diff_source": "解析に使う値",
        "diff_source_raw": "生データ（raw）",
        "diff_source_display": "表示用変換後（正規化/平滑化/単位）",
        "group_a": "グループA（基準）",
        "group_b": "グループB（比較）",
        "diff_mean_a": "A平均",
        "diff_mean_b": "B平均",
        "diff_mean_diff": "平均との差（B − A）",
        "diff_rank": "寄与が大きい波長（上位）",
        "rank_by": "ランキング基準",
        "rank_abs_diff": "|差|（平均との差の絶対値）",
        "rank_effect": "|効果量|（Cohen's d）",
        "warn_pick_groups": "グループA/Bそれぞれ1件以上選んでください。",
        "warn_effect_need_2": "効果量は各グループ2件以上で安定します（平均との差は1件でも表示します）。",
        "tab_reference": "リファレンス",
        "reference_title": "リファレンス",
        "wavelength_guide_title": "波長ガイド（主に見えるもの／感度がある物質）",
        "wavelength_guide_desc": "各バンドで一般的に観察されやすい対象の例です（用途/環境で変動します）。",
        "color_guide_title": "波長の色（参考）",
        "color_guide_desc": "波長(nm)から近似的に可視色へ変換した参考色です（反射率の色そのものではありません）。",
        "color_converter_title": "カラーコード→RGB表示",
        "color_converter_desc": "HexやRGBは即表示できます。PANTONE/DICは公式の対応表が必要なため、手元の対応表CSVをアップロードして参照します。",
        "color_system": "コード体系",
        "color_system_hex": "RGB Hex（#RRGGBB）",
        "color_system_rgb": "RGB（R,G,B）",
        "color_system_pantone": "PANTONE（要対応表）",
        "color_system_dic": "DIC（要対応表）",
        "color_code": "コード",
        "color_map_upload": "対応表CSVをアップロード（任意）",
        "color_map_template": "対応表テンプレートをダウンロード",
        "color_map_needed": "このコード体系は対応表が必要です（CSVをアップロードしてください）。",
        "color_not_found": "対応表に該当コードが見つかりませんでした。",
        "color_parse_error": "コードの形式を解釈できませんでした。",
        "color_rgb": "RGB",
        "color_hex": "Hex",
        "color_preview": "プレビュー",
        "tab_spectrum": "スペクトル",
        "tab_compare": "比較",
        "tab_pca": "PCA",
        "tab_clustering": "クラスタリング",
        "tab_boxplot": "BoxPlot",
        "tab_export": "エクスポート",
        "spectrum_single": "スペクトル（単体）",
        "select_one_sample": "表示するサンプルを1件選択",
        "band_intensity": "バンド強度（棒グラフ）",
        "wavelength_intensity": "波長別強度",
        "radar_chart": "レーダーチャート",
        "power_dial": "Power Dial",
        "ndvi_score": "NDVIスコア (0–100)",
        "state": "状態",
        "ndvi_index": "NDVI / 指標",
        "index": "指標",
        "msg_ndvi_unavailable": "NDVIを計算できません（680/810nmが必要）",
        "compare_multi": "比較（複数データ）",
        "mean_std": "平均 ± 標準偏差",
        "mean_std_help": "平均スペクトルとばらつき（±1σ）を同時に表示し、全体傾向と安定性（ばらつきの大きい波長）を把握します。",
        "mean": "平均",
        "band_mean_std": "バンド別 平均と標準偏差",
        "wavelength_nm": "波長 (nm)",
        "reflectance": "反射率",
        "radar_compare": "レーダー比較（最大5件）",
        "samples_to_compare": "比較するサンプル",
        "multi_sample_compare": "複数サンプル比較",
        "power_dial_list": "Power Dial 一覧",
        "col_sample": "サンプル",
        "col_ndvi": "NDVI",
        "col_state": "状態の目安",
        "pca_title": "主成分分析（PCA）",
        "pca_help": "18バンドを2次元に圧縮し、サンプル間の類似/差異を俯瞰します（近い点ほどスペクトルが似ています）。",
        "warn_min_2_samples": "2件以上のサンプルが必要です。",
        "pca_2d_plot": "PCA 2次元プロット",
        "group": "グループ",
        "explained_ratio": "説明率",
        "clustering_title": "クラスタリング",
        "clustering_help": "スペクトル形状の近いサンプルを自動でグルーピングします（似たもの同士のまとまりを探索）。",
        "n_clusters": "クラスタ数",
        "clustering_result": "クラスタリング結果",
        "cluster": "クラスタ",
        "boxplot_title": "箱ひげ図",
        "wavelength_nm_col": "波長(nm)",
        "box_distribution": "バンド別 分布・外れ値",
        "boxplot_help": "箱ひげ図は分布（中央値・ばらつき）と外れ値を可視化し、波長ごとの安定性や異常値を把握するのに有用です。",
        "export_title": "エクスポート",
        "export_caption": "現在のフィルタで表示しているサンプルのデータをダウンロードできます。",
        "btn_download_csv": "CSVをダウンロード",
        "info_no_data": "左サイドバーからデータをアップロードするか、サンプルデータを読み込んでください。列に 410, 435, … 940 (nm) のバンドが含まれるCSVが必要です。",
        "warn_cannot_2d": "2次元に圧縮できません。",
        # NDVI interpretations (heading, description)
        "ndvi_na": ("—", "計算できません。"),
        "ndvi_water": ("水・雪・雲など", "NDVIが負の値です。水面・雪・雲・人工物など、植被がほぼない状態と考えられます。"),
        "ndvi_bare": ("土壌・裸地・枯れ", "植被がほとんどない、または枯れた状態です。裸地、砂、岩、乾燥した地表などが該当します。"),
        "ndvi_very_low": ("ごく少ない植被", "植被はごくわずかです。荒地やストレスが強い植物の可能性があります。"),
        "ndvi_sparse": ("まばらな植被", "草地・低木・まばらな作物など、植被が中程度以下の状態です。"),
        "ndvi_moderate": ("中程度の植被", "農地や草地など、植被が安定している状態です。健全な作物・草の可能性が高いです。"),
        "ndvi_active": ("活発な植被", "植被がよく茂り、光合成が活発な状態です。健全な植物・作物と考えられます。"),
        "ndvi_dense": ("非常に密な植被", "森林や密生した作物など、植被が非常に多い状態です。生育が旺盛と考えられます。"),
    },
    "en": {
        "app_subtitle": "Visualize and explore irodori 18-band spectral data",
        "page_title": "ANSWER Lite — irodori Visualization",
        "sidebar_dataset": "Dataset",
        "upload_csv": "Upload CSV",
        "err_not_irodori": "This CSV is not in irodori format.",
        "err_bands_missing": "Missing wavelength columns (410–940 nm).",
        "btn_load_sample": "Load sample data",
        "msg_loaded": "Loaded",
        "or_load_sample": "Or click \"Load sample data\"",
        "sidebar_samples_filter": "Samples & filters",
        "hint_load_first": "Load data to enable selection",
        "sidebar_sample_select": "Sample selection",
        "include_samples": "Samples to include",
        "sidebar_filter": "Filter",
        "filter_by_label": "Filter by label",
        "filter_by_target": "Filter by target",
        "all": "(All)",
        "sidebar_pages": "Pages",
        "hint_tabs": "Switch via main tabs",
        "kpi_samples": "Samples",
        "kpi_labels": "Labels",
        "kpi_selected": "Selected",
        "kpi_status": "Status",
        "status_ready": "Ready",
        "status_no_data": "No data",
        "viz_params": "Visualization parameters",
        "norm": "Normalization",
        "norm_none": "None",
        "norm_minmax": "Min-Max (0–1)",
        "norm_zscore": "Z-score",
        "smooth": "Smoothing",
        "smooth_none": "None",
        "smooth_3": "3-point moving average",
        "smooth_5": "5-point moving average",
        "display_unit": "Display unit",
        "unit_raw": "Reflectance (raw)",
        "unit_pct": "Percentage (%)",
        "viz_note_ndvi": "NDVI and Power Dial use raw data.",
        "op_help_title": "Operation notes",
        "op_help_norm": "Normalization aligns scale across samples to compare spectral shapes more easily.",
        "op_help_norm_minmax": "Min-Max: scales each wavelength to 0–1 (useful to match display ranges).",
        "op_help_norm_zscore": "Z-score: standardizes each wavelength by mean/std (highlights deviation strength).",
        "op_help_smooth": "Smoothing (moving average) reduces noise to reveal the overall spectral trend.",
        "op_help_plot": "Bar = compare discrete band intensities / Line = compare spectral shape continuously.",
        "plot_type": "Plot type",
        "plot_bar": "Bar",
        "plot_line": "Line",
        "plot_line_markers": "Line + markers",
        "overlay_title": "Overlay (multiple samples)",
        "overlay_hint": "Overlay multiple samples on the same axis for direct shape comparison.",
        "max_overlay": "Max traces",
        "diff_tab": "Difference",
        "diff_title": "Spectral difference analysis",
        "diff_desc": "Compute mean spectrum differences between two groups and rank wavelengths by difference magnitude (e.g., effect size).",
        "diff_source": "Values for analysis",
        "diff_source_raw": "Raw data",
        "diff_source_display": "Display-transformed (norm/smooth/unit)",
        "group_a": "Group A (baseline)",
        "group_b": "Group B (comparison)",
        "diff_mean_a": "Mean A",
        "diff_mean_b": "Mean B",
        "diff_mean_diff": "Mean difference (B − A)",
        "diff_rank": "Top contributing wavelengths",
        "rank_by": "Rank by",
        "rank_abs_diff": "|diff| (absolute mean difference)",
        "rank_effect": "|effect| (Cohen's d)",
        "warn_pick_groups": "Please choose at least one sample for both Group A and Group B.",
        "warn_effect_need_2": "Effect size is more stable with at least 2 samples per group (mean difference will still be shown).",
        "tab_reference": "Reference",
        "reference_title": "Reference",
        "wavelength_guide_title": "Wavelength guide (what is typically visible/sensitive)",
        "wavelength_guide_desc": "Examples of targets commonly observed per band (may vary by application and environment).",
        "color_guide_title": "Wavelength colors (reference)",
        "color_guide_desc": "Approximate visible colors converted from wavelength (nm). This is not the literal color of reflectance values.",
        "color_converter_title": "Color code → RGB preview",
        "color_converter_desc": "Hex/RGB can be previewed immediately. PANTONE/DIC require a lookup table; upload your mapping CSV to reference them.",
        "color_system": "Code system",
        "color_system_hex": "RGB Hex (#RRGGBB)",
        "color_system_rgb": "RGB (R,G,B)",
        "color_system_pantone": "PANTONE (needs mapping)",
        "color_system_dic": "DIC (needs mapping)",
        "color_code": "Code",
        "color_map_upload": "Upload mapping CSV (optional)",
        "color_map_template": "Download mapping template",
        "color_map_needed": "This code system requires a mapping CSV (please upload one).",
        "color_not_found": "Code not found in the mapping.",
        "color_parse_error": "Could not parse the code format.",
        "color_rgb": "RGB",
        "color_hex": "Hex",
        "color_preview": "Preview",
        "tab_spectrum": "Spectrum",
        "tab_compare": "Compare",
        "tab_pca": "PCA",
        "tab_clustering": "Clustering",
        "tab_boxplot": "Box Plot",
        "tab_export": "Export",
        "spectrum_single": "Spectrum (single)",
        "select_one_sample": "Select one sample to display",
        "band_intensity": "Band intensity (bar)",
        "wavelength_intensity": "Intensity by wavelength",
        "radar_chart": "Radar chart",
        "power_dial": "Power Dial",
        "ndvi_score": "NDVI score (0–100)",
        "state": "State",
        "ndvi_index": "NDVI / Index",
        "index": "Index",
        "msg_ndvi_unavailable": "NDVI cannot be calculated (680/810 nm required).",
        "compare_multi": "Compare (multiple)",
        "mean_std": "Mean ± Std dev",
        "mean_std_help": "Shows the mean spectrum with variability (±1σ) to understand overall trends and which wavelengths are unstable (high variance).",
        "mean": "Mean",
        "band_mean_std": "Mean and std dev by band",
        "wavelength_nm": "Wavelength (nm)",
        "reflectance": "Reflectance",
        "radar_compare": "Radar comparison (max 5)",
        "samples_to_compare": "Samples to compare",
        "multi_sample_compare": "Multiple sample comparison",
        "power_dial_list": "Power Dial list",
        "col_sample": "Sample",
        "col_ndvi": "NDVI",
        "col_state": "State",
        "pca_title": "Principal Component Analysis (PCA)",
        "pca_help": "Compresses 18 bands into 2D to view similarity/differences across samples (closer points indicate more similar spectra).",
        "warn_min_2_samples": "At least 2 samples are required.",
        "pca_2d_plot": "PCA 2D plot",
        "group": "Group",
        "explained_ratio": "Explained variance",
        "clustering_title": "Clustering",
        "clustering_help": "Automatically groups samples with similar spectral shapes to help discover natural clusters.",
        "n_clusters": "Number of clusters",
        "clustering_result": "Clustering result",
        "cluster": "Cluster",
        "boxplot_title": "Box plot",
        "wavelength_nm_col": "Wavelength (nm)",
        "box_distribution": "Distribution and outliers by band",
        "boxplot_help": "Box plots show distribution (median/spread) and outliers, useful for spotting unstable bands or anomalies.",
        "export_title": "Export",
        "export_caption": "Download data for the samples currently shown (after filters).",
        "btn_download_csv": "Download CSV",
        "info_no_data": "Upload a CSV from the sidebar or load sample data. The CSV must include wavelength columns 410, 435, … 940 (nm).",
        "warn_cannot_2d": "Cannot reduce to 2 dimensions.",
        "ndvi_na": ("—", "Cannot calculate."),
        "ndvi_water": ("Water / snow / clouds", "Negative NDVI. Likely water, snow, clouds, or artificial surfaces with little vegetation."),
        "ndvi_bare": ("Bare soil / barren", "Little or no vegetation. Bare soil, sand, rock, or dry surface."),
        "ndvi_very_low": ("Very low vegetation", "Very sparse vegetation. Possibly stressed plants or barren land."),
        "ndvi_sparse": ("Sparse vegetation", "Grassland, shrubs, or sparse crops."),
        "ndvi_moderate": ("Moderate vegetation", "Stable vegetation such as cropland or grassland. Likely healthy crops or grass."),
        "ndvi_active": ("Active vegetation", "Dense vegetation with active photosynthesis. Likely healthy plants or crops."),
        "ndvi_dense": ("Dense vegetation", "Forest or dense crops. Vigorous growth."),
    },
}

# =============================================================================
# 波長ごとのガイド（一般的な例）
# =============================================================================
WAVELENGTH_GUIDE = {
    410: {
        "ja": "ヘモグロビン（Soret band）、ポルフィリン、血液、細胞核染色（H&E）",
        "en": "Hemoglobin (Soret band), porphyrins, blood, nuclear staining (H&E)",
    },
    435: {"ja": "ヘモグロビン、青色顔料、メラニン", "en": "Hemoglobin, blue pigments, melanin"},
    460: {"ja": "カロテノイド、フラビン、細菌蛍光", "en": "Carotenoids, flavins, bacterial fluorescence"},
    485: {"ja": "NADH、フラビン、植物の光合成補助色素", "en": "NADH, flavins, photosynthetic accessory pigments"},
    510: {"ja": "クロロフィルb、植物色素、細胞組織", "en": "Chlorophyll b, plant pigments, tissue structure"},
    535: {"ja": "オキシヘモグロビン吸収ピーク", "en": "Oxyhemoglobin absorption peak"},
    560: {"ja": "デオキシヘモグロビン、血流評価", "en": "Deoxyhemoglobin, blood flow assessment"},
    585: {"ja": "血液吸収、皮膚の血管", "en": "Blood absorption, skin vasculature"},
    610: {"ja": "ポルフィリン、腫瘍関連蛍光", "en": "Porphyrins, tumor-associated fluorescence"},
    645: {"ja": "クロロフィルa吸収帯", "en": "Chlorophyll a absorption band"},
    680: {"ja": "クロロフィルa強吸収（光合成）", "en": "Strong chlorophyll a absorption (photosynthesis)"},
    705: {"ja": "レッドエッジ：植物健康状態", "en": "Red edge: plant health condition"},
    730: {"ja": "クロロフィル蛍光", "en": "Chlorophyll fluorescence"},
    760: {"ja": "O2吸収帯（大気）、植物ストレス", "en": "O2 absorption band (atmosphere), plant stress"},
    810: {"ja": "生体組織透過、血管観察", "en": "Tissue penetration, vascular observation"},
    860: {"ja": "水分・細胞構造", "en": "Water content, cellular structure"},
    900: {"ja": "有機物、水分、脂質", "en": "Organics, water content, lipids"},
    940: {"ja": "水の強い吸収帯", "en": "Strong water absorption band"},
}


def wavelength_desc_nm(wavelength_nm: int) -> str:
    lang = _lang()
    d = WAVELENGTH_GUIDE.get(int(wavelength_nm))
    if not d:
        return ""
    return d.get(lang, d.get("ja", ""))


def wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8):
    """Approximate wavelength(nm) -> sRGB (0-255). Based on common heuristic mapping (380-780nm)."""
    wl = float(wavelength_nm)
    if wl < 380 or wl > 780:
        return 160, 160, 160
    if wl < 440:
        r, g, b = -(wl - 440) / (440 - 380), 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / (490 - 440), 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / (510 - 490)
    elif wl < 580:
        r, g, b = (wl - 510) / (580 - 510), 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / (645 - 580), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl < 701:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    def _c(v):
        if v <= 0:
            return 0
        return int(round(255 * ((v * factor) ** gamma)))
    return _c(r), _c(g), _c(b)


def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def parse_hex_color(s: str):
    if not s:
        return None
    v = s.strip()
    if v.startswith("#"):
        v = v[1:]
    if len(v) != 6:
        return None
    try:
        r = int(v[0:2], 16)
        g = int(v[2:4], 16)
        b = int(v[4:6], 16)
        return r, g, b
    except Exception:
        return None


def parse_rgb_triplet(s: str):
    if not s:
        return None
    v = s.strip().replace(" ", "")
    parts = v.split(",")
    if len(parts) != 3:
        return None
    try:
        r, g, b = (int(p) for p in parts)
        if any(x < 0 or x > 255 for x in (r, g, b)):
            return None
        return r, g, b
    except Exception:
        return None


def render_color_preview(rgb):
    hx = rgb_to_hex(rgb)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;">
          <div style="width:44px;height:20px;border-radius:6px;border:1px solid #e5e7eb;background:{hx};"></div>
          <div style="font-size:0.9rem;"><b>{t('color_hex')}:</b> {hx} &nbsp; <b>{t('color_rgb')}:</b> {rgb}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _lang():
    return st.session_state.get("lang", "ja")


def t(key):
    """現在言語の文字列を返す。"""
    lang = _lang()
    if key not in TRANSLATIONS.get(lang, TRANSLATIONS["ja"]):
        return TRANSLATIONS["ja"].get(key, key)
    val = TRANSLATIONS[lang][key]
    return val


def get_ndvi_interpretation_i18n(ndvi):
    """NDVIの値から状態の解釈を返す（日英）。"""
    lang = _lang()
    D = TRANSLATIONS[lang]
    if np.isnan(ndvi):
        return D["ndvi_na"][0], D["ndvi_na"][1]
    v = float(ndvi)
    if v < -0.1:
        return D["ndvi_water"][0], D["ndvi_water"][1]
    if v < 0.1:
        return D["ndvi_bare"][0], D["ndvi_bare"][1]
    if v < 0.2:
        return D["ndvi_very_low"][0], D["ndvi_very_low"][1]
    if v < 0.4:
        return D["ndvi_sparse"][0], D["ndvi_sparse"][1]
    if v < 0.6:
        return D["ndvi_moderate"][0], D["ndvi_moderate"][1]
    if v < 0.8:
        return D["ndvi_active"][0], D["ndvi_active"][1]
    return D["ndvi_dense"][0], D["ndvi_dense"][1]

# =============================================================================
# カスタムCSS（SaaS風）
# =============================================================================
def inject_css():
    st.markdown("""
    <style>
    /* 全体 */
    .stApp { max-width: 100%; }
    .main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

    /* 見出し階層 */
    h1 { font-size: 1.5rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.25rem; }
    h2 { font-size: 1.125rem; font-weight: 600; color: #1a1a2e; margin-top: 1.25rem; margin-bottom: 0.5rem; }
    h3 { font-size: 0.9375rem; font-weight: 500; color: #4a4a6a; margin-top: 0.75rem; margin-bottom: 0.35rem; }

    /* アクセント色 */
    :root { --accent: #2563eb; --accent-light: #eff6ff; }

    /* KPIカード */
    .kpi-card {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 0.5rem;
    }
    .kpi-card .kpi-value { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
    .kpi-card .kpi-label { font-size: 0.75rem; color: #6b7280; margin-top: 0.15rem; }

    /* セクションカード */
    .section-card {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    /* メッセージ整理 */
    [data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    [data-testid="stAlert"] div[role="alert"] {
        border-radius: 6px;
    }

    /* サイドバー */
    [data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] .stMarkdown { font-size: 0.875rem; }

    /* 入力統一 */
    .stSelectbox, .stMultiSelect, .stSlider { width: 100%; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# データ読み込み・検証（既存ロジック）
# =============================================================================
def load_irodori_csv(uploaded_file):
    """アップロードされたCSVを読み込み、バンド列を検証する。"""
    df = pd.read_csv(uploaded_file)
    col_set = {str(c).strip() for c in df.columns}
    bands = [c for c in BAND_COLS if c in col_set]
    if len(bands) < 2:
        return None, []  # 呼び出し元でエラー表示
    band_cols_actual = [c for c in df.columns if str(c).strip() in bands]
    band_cols_actual.sort(key=lambda x: WAVELENGTHS[BAND_COLS.index(str(x).strip())])
    return df, band_cols_actual


def get_band_data(df, bands):
    """バンド列のみのDataFrame。数値に変換。"""
    out = df[bands].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def make_row_label(df, idx):
    """1行の表示用ラベル。"""
    row = df.iloc[idx]
    parts = []
    if "target" in df.columns:
        parts.append(str(row.get("target", "")))
    if "label" in df.columns:
        parts.append(str(row.get("label", "")))
    if "device_timestamp" in df.columns:
        parts.append(str(row.get("device_timestamp", ""))[:19])
    elif "server_timestamp" in df.columns:
        parts.append(str(row.get("server_timestamp", ""))[:19])
    return " / ".join(parts) if parts else f"Row {idx}"


def make_row_label_by_pos(df, pos):
    """フィルタ後のDataFrameの行位置posに対する表示ラベル（元dfの情報を使う場合は呼び出し元で対応）。"""
    row = df.iloc[pos]
    parts = []
    if "target" in df.columns:
        parts.append(str(row.get("target", "")))
    if "label" in df.columns:
        parts.append(str(row.get("label", "")))
    if "device_timestamp" in df.columns:
        parts.append(str(row.get("device_timestamp", ""))[:19])
    elif "server_timestamp" in df.columns:
        parts.append(str(row.get("server_timestamp", ""))[:19])
    return " / ".join(parts) if parts else f"Row {pos}"


# =============================================================================
# 可視化用データ変換（正規化・平滑化）
# =============================================================================
def apply_display_transform(band_data, bands, norm_type, smooth_window, display_unit):
    """表示用に正規化・平滑化・単位変換のみ適用。NDVI計算には使わずチャート表示用。"""
    X = band_data[bands].fillna(0).copy()
    if smooth_window and smooth_window > 1:
        k = smooth_window
        for i in range(len(X)):
            row = X.iloc[i].values
            smoothed = np.convolve(row, np.ones(k) / k, mode="same")
            X.iloc[i] = smoothed
    if norm_type in ("Min-Max (0–1)",):
        for c in X.columns:
            mn, mx = X[c].min(), X[c].max()
            if mx > mn:
                X[c] = (X[c] - mn) / (mx - mn)
    elif norm_type in ("Z-score",):
        for c in X.columns:
            mu, sig = X[c].mean(), X[c].std()
            if sig and sig > 0:
                X[c] = (X[c] - mu) / sig
    if display_unit in ("百分率 (%)", "Percentage (%)"):
        for c in X.columns:
            mx = X[c].max()
            if mx and mx > 0:
                X[c] = X[c] / mx * 100
    return X


# =============================================================================
# チャート生成（既存ロジック維持・データだけ差し替え可能）
# =============================================================================
def single_bar_graph(series, bands, title=None):
    if title is None:
        title = t("band_intensity")
    x = [int(b) for b in bands]
    hover_text = [wavelength_desc_nm(xx) for xx in x]
    colors = [rgb_to_hex(wavelength_to_rgb(xx)) for xx in x]
    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=series.values,
                marker_color=colors,
                text=hover_text,
                hovertemplate="λ: %{x} nm<br>y: %{y}<br>%{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=t("wavelength_nm"),
        yaxis_title=t("reflectance"),
        template="plotly_white",
        height=380,
        margin=dict(t=40, b=50),
    )
    fig.update_xaxes(type="linear")
    return fig


def single_line_graph(series, bands, title=None, with_markers=True):
    if title is None:
        title = t("wavelength_intensity")
    x = [int(b) for b in bands]
    mode = "lines+markers" if with_markers else "lines"
    hover_texts = [wavelength_desc_nm(xx) for xx in x]
    fig = go.Figure(data=[go.Scatter(x=x, y=series.values, mode=mode)])
    fig.update_traces(text=hover_texts, hovertemplate="λ: %{x} nm<br>y: %{y}<br>%{text}<extra></extra>")
    fig.update_layout(
        title=title,
        xaxis_title=t("wavelength_nm"),
        yaxis_title=t("reflectance"),
        template="plotly_white",
        height=380,
        margin=dict(t=40, b=50),
    )
    fig.update_xaxes(type="linear")
    return fig


def overlay_line_graph(df, band_data, bands, positions, title, with_markers=True, max_traces=12):
    x = [int(b) for b in bands]
    mode = "lines+markers" if with_markers else "lines"
    hover_texts = [wavelength_desc_nm(xx) for xx in x]
    fig = go.Figure()
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Dark24
    for k, pos in enumerate(positions[:max_traces]):
        row = band_data.iloc[pos]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=row.values,
                mode=mode,
                name=make_row_label_by_pos(df, pos),
                line=dict(color=colors[k % len(colors)], width=2),
                text=hover_texts,
                hovertemplate="λ: %{x} nm<br>y: %{y}<br>%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=t("wavelength_nm"),
        yaxis_title=t("reflectance"),
        template="plotly_white",
        height=420,
        margin=dict(t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(type="linear")
    return fig


def mean_std_line_graph(mean_vals, std_vals, bands, title=None):
    if title is None:
        title = t("band_mean_std")
    x = [int(b) for b in bands]
    upper = (mean_vals + std_vals).values
    lower = (mean_vals - std_vals).values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="±1σ",
            hoverinfo="skip",
        )
    )
    hover_texts = [wavelength_desc_nm(xx) for xx in x]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean_vals.values,
            mode="lines+markers",
            name=t("mean"),
            line=dict(width=2),
            text=hover_texts,
            hovertemplate="λ: %{x} nm<br>mean: %{y}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=t("wavelength_nm"),
        yaxis_title=t("reflectance"),
        template="plotly_white",
        height=380,
        margin=dict(t=40, b=50),
    )
    fig.update_xaxes(type="linear")
    return fig


def spectral_diff_rank(mean_a, mean_b, std_a, std_b, n_a, n_b, bands):
    """波長ごとの差分指標を作る。Cohen's dとWelch風tスコア（p値なし）を併記。"""
    diff = (mean_b - mean_a).astype(float)
    pooled = np.full_like(diff.values, np.nan, dtype=float)
    if n_a >= 2 and n_b >= 2:
        s2a = (std_a.astype(float) ** 2).values
        s2b = (std_b.astype(float) ** 2).values
        denom = (n_a + n_b - 2)
        if denom > 0:
            pooled_var = ((n_a - 1) * s2a + (n_b - 1) * s2b) / denom
            pooled = np.sqrt(np.maximum(pooled_var, 0))
    d = diff.values / pooled
    welch = diff.values / np.sqrt((std_a.astype(float).values ** 2) / max(n_a, 1) + (std_b.astype(float).values ** 2) / max(n_b, 1))
    out = pd.DataFrame(
        {
            "wavelength_nm": [int(b) for b in bands],
            "diff": diff.values,
            "abs_diff": np.abs(diff.values),
            "cohens_d": d,
            "abs_cohens_d": np.abs(d),
            "welch_score": welch,
            "abs_welch_score": np.abs(welch),
        }
    )
    return out


def single_radar_chart(series, bands):
    values = series.values.tolist() + [series.values[0]]
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=bands + [bands[0]],
        fill="toself",
        name=t("reflectance"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=t("radar_chart"),
        showlegend=False,
        template="plotly_white",
        height=420,
    )
    return fig


def power_dial_gauge(value, title=None, min_val=0, max_val=100):
    if title is None:
        title = t("ndvi_score")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        title={"text": title},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "var(--accent, #2563eb)"},
            "steps": [
                {"range": [min_val, (max_val - min_val) * 0.33 + min_val], "color": "#f3f4f6"},
                {"range": [(max_val - min_val) * 0.33 + min_val, (max_val - min_val) * 0.66 + min_val], "color": "#fef3c7"},
                {"range": [(max_val - min_val) * 0.66 + min_val, max_val], "color": "#d1fae5"},
            ],
            "threshold": {
                "line": {"color": "#dc2626", "width": 3},
                "thickness": 0.75,
                "value": (max_val - min_val) * 0.8 + min_val,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def calc_ndvi(row_bands, red_col, nir_col):
    red = row_bands.get(red_col, np.nan)
    nir = row_bands.get(nir_col, np.nan)
    if pd.isna(red) or pd.isna(nir):
        return np.nan
    s = red + nir
    if s == 0:
        return 0.0
    return (float(nir) - float(red)) / s


def calc_spectral_index(row_bands, bands, index_type="NDVI"):
    if index_type == "NDVI":
        return calc_ndvi(row_bands, RED_BAND, NIR_BAND)
    if index_type in ("Simple Ratio (NIR/Red)",):
        red = row_bands.get(RED_BAND, np.nan)
        nir = row_bands.get(NIR_BAND, np.nan)
        if pd.isna(red) or red == 0:
            return np.nan
        return float(nir) / float(red)
    if index_type in ("Total Reflectance",):
        return row_bands[bands].sum()
    return np.nan


# =============================================================================
# サイドバー：言語・データ選択・フィルタ・ナビ
# =============================================================================
def render_sidebar():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ja"
    st.sidebar.selectbox(
        "Language / 言語",
        options=["ja", "en"],
        format_func=lambda x: "日本語" if x == "ja" else "English",
        key="lang",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### " + t("sidebar_dataset"))
    _dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(_dir, "irodori-sample-data.csv")
    use_sample = os.path.exists(default_path)

    uploaded = st.sidebar.file_uploader(t("upload_csv"), type=["csv"], key="upload", label_visibility="collapsed")
    if uploaded is not None:
        df, bands = load_irodori_csv(uploaded)
        if df is not None and len(bands) >= 2:
            st.session_state["df"] = df
            st.session_state["bands"] = bands
            st.session_state["data_source"] = "uploaded"
        else:
            if "df" in st.session_state:
                st.sidebar.error(t("err_not_irodori"))
            else:
                st.sidebar.error(t("err_bands_missing"))
    elif use_sample and st.sidebar.button(t("btn_load_sample"), use_container_width=True):
        with open(default_path, "rb") as f:
            df, bands = load_irodori_csv(io.BytesIO(f.read()))
        if df is not None:
            st.session_state["df"] = df
            st.session_state["bands"] = bands
            st.session_state["data_source"] = "sample"
            st.sidebar.success(t("msg_loaded"))
    elif use_sample:
        st.sidebar.caption(t("or_load_sample"))

    if "df" not in st.session_state or "bands" not in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### " + t("sidebar_samples_filter"))
        st.sidebar.caption(t("hint_load_first"))
        return None, None, [], []

    df = st.session_state["df"]
    bands = st.session_state["bands"]
    band_data = get_band_data(df, bands)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### " + t("sidebar_sample_select"))
    options = [make_row_label(df, i) for i in range(len(df))]
    default_idx = list(range(len(df)))
    selected_indices = st.sidebar.multiselect(
        t("include_samples"),
        range(len(df)),
        default=default_idx,
        format_func=lambda i: options[i],
        key="sidebar_samples",
    )
    if not selected_indices:
        selected_indices = list(range(len(df)))

    st.sidebar.markdown("### " + t("sidebar_filter"))
    filter_label = None
    all_label = t("all")
    if "label" in df.columns:
        labels = df["label"].dropna().astype(str).unique().tolist()
        if labels:
            filter_label = st.sidebar.selectbox(t("filter_by_label"), [all_label] + labels, key="filter_label")
    filter_target = None
    if "target" in df.columns:
        targets = df["target"].dropna().astype(str).unique().tolist()
        if targets:
            filter_target = st.sidebar.selectbox(t("filter_by_target"), [all_label] + targets, key="filter_target")

    filtered_idx = selected_indices
    if filter_label and filter_label != all_label:
        filtered_idx = [i for i in filtered_idx if str(df.iloc[i].get("label", "")) == filter_label]
    if filter_target and filter_target != all_label:
        filtered_idx = [i for i in filtered_idx if str(df.iloc[i].get("target", "")) == filter_target]
    if not filtered_idx:
        filtered_idx = selected_indices

    st.sidebar.markdown("---")
    st.sidebar.markdown("### " + t("sidebar_pages"))
    st.sidebar.caption(t("hint_tabs"))

    df_f = df.iloc[filtered_idx].reset_index(drop=True)
    band_data_f = band_data.iloc[filtered_idx].reset_index(drop=True)
    # 元のdfのインデックスを保持したリスト（ラベル表示用）
    index_map = filtered_idx  # 元dfのidxのリスト

    return df_f, band_data_f, bands, index_map


# =============================================================================
# KPIカード
# =============================================================================
def render_kpis(df, band_data, bands, index_map, data_source):
    n_samples = len(df)
    n_labels = df["label"].nunique() if "label" in df.columns else 0
    n_selected = len(index_map)
    status = t("status_ready") if n_samples > 0 else t("status_no_data")

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, label) in zip([c1, c2, c3, c4], [
        (n_samples, t("kpi_samples")),
        (n_labels if n_labels else "—", t("kpi_labels")),
        (n_selected, t("kpi_selected")),
        (status, t("kpi_status")),
    ]):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)


# =============================================================================
# 右パネル（Expander）：可視化パラメータ
# =============================================================================
def get_viz_params():
    with st.expander(t("viz_params"), expanded=False):
        norm_labels = [t("norm_none"), t("norm_minmax"), t("norm_zscore")]
        norm_internal = ["なし", "Min-Max (0–1)", "Z-score"]
        norm_idx = st.selectbox(t("norm"), range(3), format_func=lambda i: norm_labels[i], key="viz_norm")
        norm_type = norm_internal[norm_idx]
        with st.container():
            st.caption(t("op_help_norm"))
            if norm_type == "Min-Max (0–1)":
                st.caption(t("op_help_norm_minmax"))
            elif norm_type == "Z-score":
                st.caption(t("op_help_norm_zscore"))
        smooth_labels = [t("smooth_none"), t("smooth_3"), t("smooth_5")]
        smooth_internal = [None, 3, 5]
        smooth_idx = st.selectbox(t("smooth"), range(3), format_func=lambda i: smooth_labels[i], key="viz_smooth")
        smooth_window = smooth_internal[smooth_idx]
        st.caption(t("op_help_smooth"))
        unit_labels = [t("unit_raw"), t("unit_pct")]
        unit_internal = ["反射率（そのまま）", "百分率 (%)"]
        unit_idx = st.selectbox(t("display_unit"), range(2), format_func=lambda i: unit_labels[i], key="viz_unit")
        display_unit = unit_internal[unit_idx]
        st.caption(t("viz_note_ndvi"))
    return norm_type, smooth_window, display_unit


# =============================================================================
# タブ：Spectrum（単体）
# =============================================================================
def tab_spectrum(df, band_data_raw, band_data_display, bands, index_map):
    st.markdown("### " + t("spectrum_single"))
    options = [make_row_label_by_pos(df, i) for i in range(len(df))]
    selected_pos = st.selectbox(
        t("select_one_sample"),
        range(len(df)),
        format_func=lambda i: options[i],
        key="spectrum_select",
    )
    row_raw = band_data_raw.iloc[selected_pos]
    row_display = band_data_display.iloc[selected_pos]
    label = options[selected_pos]

    col_chart, col_params = st.columns([3, 1])
    with col_chart:
        st.markdown("#### " + t("band_intensity"))
        plot_labels = [t("plot_bar"), t("plot_line_markers")]
        plot_internal = ["bar", "line_markers"]
        plot_idx = st.radio(
            t("plot_type"),
            range(2),
            format_func=lambda i: plot_labels[i],
            horizontal=True,
            index=1,
            key="spectrum_plot_type",
        )
        st.caption(t("op_help_plot"))
        mode = plot_internal[plot_idx]
        if mode == "bar":
            fig_main = single_bar_graph(row_display, bands, title=f"{t('wavelength_intensity')} — {label}")
        else:
            fig_main = single_line_graph(row_display, bands, title=f"{t('wavelength_intensity')} — {label}", with_markers=True)
        st.plotly_chart(
            fig_main,
            use_container_width=True,
            key="tab_spectrum_bar",
        )
        st.markdown("#### " + t("radar_chart"))
        st.plotly_chart(
            single_radar_chart(row_display, bands),
            use_container_width=True,
            key="tab_spectrum_radar",
        )

    with col_params:
        st.markdown("#### " + t("power_dial"))
        ndvi = calc_ndvi(row_raw, RED_BAND, NIR_BAND)
        if not np.isnan(ndvi):
            score_0_100 = (ndvi + 1) / 2 * 100
            st.plotly_chart(
                power_dial_gauge(round(score_0_100, 1), title=t("ndvi_score"), min_val=0, max_val=100),
                use_container_width=True,
                key="tab_spectrum_gauge",
            )
            heading, desc = get_ndvi_interpretation_i18n(ndvi)
            st.markdown(f"**{t('state')}:** {heading}")
            st.caption(desc)
        else:
            st.info(t("msg_ndvi_unavailable"))

        st.markdown("#### " + t("ndvi_index"))
        index_options = ["NDVI", "Simple Ratio (NIR/Red)", "Total Reflectance"]
        index_type = st.selectbox(t("index"), index_options, key="spectrum_index")
        val = calc_spectral_index(row_raw, bands, index_type)
        if not np.isnan(val):
            st.metric(index_type, f"{val:.4f}")
            if index_type == "NDVI":
                h, d = get_ndvi_interpretation_i18n(val)
                st.caption(f"{h} — {d[:50]}…")


# =============================================================================
# タブ：Compare（複数比較）
# =============================================================================
def tab_compare(df, band_data_raw, band_data_display, bands, index_map):
    st.markdown("### " + t("compare_multi"))

    st.markdown("#### " + t("mean_std"))
    st.caption(t("mean_std_help"))
    mean_vals = band_data_display[bands].mean()
    std_vals = band_data_display[bands].std()
    mean_plot_labels = [t("plot_line_markers"), t("plot_bar")]
    mean_plot_internal = ["line_markers", "bar"]
    mean_plot_idx = st.radio(
        t("plot_type"),
        range(2),
        format_func=lambda i: mean_plot_labels[i],
        horizontal=True,
        index=0,
        key="compare_mean_plot_type",
    )
    mean_mode = mean_plot_internal[mean_plot_idx]
    if mean_mode == "bar":
        fig = go.Figure(
            [
                go.Bar(x=bands, y=mean_vals, name=t("mean")),
                go.Scatter(x=bands, y=mean_vals + std_vals, mode="lines", line=dict(width=0), showlegend=False),
                go.Scatter(x=bands, y=mean_vals - std_vals, mode="lines", line=dict(width=0), fill="tonexty", name="±1σ"),
            ]
        )
        fig.update_layout(
            title=t("band_mean_std"),
            xaxis_title=t("wavelength_nm"),
            yaxis_title=t("reflectance"),
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True, key="tab_compare_bar_std")
    else:
        fig = mean_std_line_graph(mean_vals, std_vals, bands, title=t("band_mean_std"))
        st.plotly_chart(fig, use_container_width=True, key="tab_compare_line_std")

    st.markdown("#### " + t("overlay_title"))
    st.caption(t("overlay_hint"))
    options = [make_row_label_by_pos(df, i) for i in range(len(df))]
    max_traces = st.slider(t("max_overlay"), 2, min(20, len(df)), min(8, len(df)), key="overlay_max")
    overlay_indices = st.multiselect(
        t("samples_to_compare"),
        range(len(df)),
        default=list(range(min(5, len(df)))),
        format_func=lambda i: options[i],
        key="compare_overlay_select",
    )
    overlay_plot_labels = [t("plot_line_markers")]
    overlay_plot_internal = ["line_markers"]
    overlay_plot_idx = st.radio(
        t("plot_type"),
        range(1),
        format_func=lambda i: overlay_plot_labels[i],
        horizontal=True,
        index=0,
        key="compare_overlay_plot_type",
    )
    if overlay_indices:
        fig_ov = overlay_line_graph(
            df,
            band_data_display[bands],
            bands,
            overlay_indices,
            title=t("overlay_title"),
            with_markers=(overlay_plot_internal[overlay_plot_idx] == "line_markers"),
            max_traces=max_traces,
        )
        st.plotly_chart(fig_ov, use_container_width=True, key="tab_compare_overlay")

    st.markdown("#### " + t("radar_compare"))
    indices = st.multiselect(
        t("samples_to_compare"),
        range(len(df)),
        default=list(range(min(3, len(df)))),
        format_func=lambda i: options[i],
        key="compare_radar_select",
    )
    if indices:
        indices = indices[:5]
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        for k, idx in enumerate(indices):
            row = band_data_display.iloc[idx]
            vals = row.values.tolist() + [row.values[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=bands + [bands[0]],
                fill="toself",
                name=options[idx],
                line=dict(color=colors[k % len(colors)]),
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title=t("multi_sample_compare"), template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True, key="tab_compare_radar")

    st.markdown("#### " + t("power_dial_list"))
    raw_ndvi = [calc_ndvi(band_data_raw.iloc[i], RED_BAND, NIR_BAND) for i in range(len(band_data_raw))]
    scores = [(v + 1) / 2 * 100 if not np.isnan(v) else 0 for v in raw_ndvi]
    n = len(scores)
    cols = st.columns(min(4, n))
    for i, (idx, sc) in enumerate(zip(range(n), scores)):
        with cols[i % len(cols)]:
            st.plotly_chart(
                power_dial_gauge(round(sc, 1), title=options[idx][:18] if i < len(options) else f"#{idx}", min_val=0, max_val=100),
                use_container_width=True,
                key=f"tab_compare_gauge_{idx}",
            )
    batch_df = pd.DataFrame([
        (options[i], f"{raw_ndvi[i]:.4f}" if not np.isnan(raw_ndvi[i]) else "—", get_ndvi_interpretation_i18n(raw_ndvi[i])[0])
        for i in range(len(raw_ndvi))
    ], columns=[t("col_sample"), t("col_ndvi"), t("col_state")])
    st.dataframe(batch_df, use_container_width=True, hide_index=True)


def tab_difference(df, band_data_raw, band_data_display, bands):
    st.markdown("### " + t("diff_title"))
    st.caption(t("diff_desc"))

    source_labels = [t("diff_source_raw"), t("diff_source_display")]
    source_internal = ["raw", "display"]
    source_idx = st.radio(t("diff_source"), range(2), format_func=lambda i: source_labels[i], horizontal=True, key="diff_source_choice")
    src = source_internal[source_idx]
    X = (band_data_raw[bands] if src == "raw" else band_data_display[bands]).copy()

    options = [make_row_label_by_pos(df, i) for i in range(len(df))]
    default_a = list(range(min(3, len(df))))
    default_b = list(range(min(3, len(df)), min(6, len(df))))
    group_a = st.multiselect(t("group_a"), range(len(df)), default=default_a, format_func=lambda i: options[i], key="diff_group_a")
    group_b = st.multiselect(t("group_b"), range(len(df)), default=default_b, format_func=lambda i: options[i], key="diff_group_b")

    if (not group_a) or (not group_b):
        st.warning(t("warn_pick_groups"))
        return

    A = X.iloc[group_a]
    B = X.iloc[group_b]
    mean_a = A.mean()
    mean_b = B.mean()
    std_a = A.std()
    std_b = B.std()
    n_a, n_b = len(A), len(B)

    if n_a < 2 or n_b < 2:
        st.info(t("warn_effect_need_2"))

    # Mean spectra overlay
    st.markdown("#### " + t("mean_std"))
    x = [int(b) for b in bands]
    hover_texts = [wavelength_desc_nm(xx) for xx in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=mean_a.values, mode="lines+markers", name=t("diff_mean_a"), line=dict(width=2), text=hover_texts, hovertemplate="λ: %{x} nm<br>A mean: %{y}<br>%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=mean_b.values, mode="lines+markers", name=t("diff_mean_b"), line=dict(width=2), text=hover_texts, hovertemplate="λ: %{x} nm<br>B mean: %{y}<br>%{text}<extra></extra>"))
    fig.update_layout(template="plotly_white", height=380, xaxis_title=t("wavelength_nm"), yaxis_title=t("reflectance"))
    fig.update_xaxes(type="linear")
    st.plotly_chart(fig, use_container_width=True, key="diff_means_overlay")

    # Mean difference
    st.markdown("#### " + t("diff_mean_diff"))
    diff = (mean_b - mean_a).astype(float)
    fig_d = go.Figure()
    fig_d.add_trace(go.Bar(x=x, y=diff.values, name=t("diff_mean_diff")))
    fig_d.add_hline(y=0, line_width=1, line_color="#9ca3af")
    fig_d.update_layout(template="plotly_white", height=340, xaxis_title=t("wavelength_nm"), yaxis_title=t("reflectance"))
    fig_d.update_xaxes(type="linear")
    st.plotly_chart(fig_d, use_container_width=True, key="diff_bar")

    # Ranking table
    st.markdown("#### " + t("diff_rank"))
    rank_labels = [t("rank_abs_diff"), t("rank_effect")]
    rank_internal = ["abs_diff", "abs_cohens_d"]
    rank_idx = st.radio(t("rank_by"), range(2), format_func=lambda i: rank_labels[i], horizontal=True, key="diff_rank_by")
    rank_key = rank_internal[rank_idx]
    rank_df = spectral_diff_rank(mean_a, mean_b, std_a, std_b, n_a, n_b, bands)
    rank_df = rank_df.sort_values(rank_key, ascending=False).head(8).reset_index(drop=True)
    show = rank_df[["wavelength_nm", "diff", "cohens_d", "welch_score"]].copy()
    show.columns = [t("wavelength_nm"), "diff", "Cohen's d", "Welch score"]
    st.dataframe(show, use_container_width=True, hide_index=True)


# =============================================================================
# タブ：PCA
# =============================================================================
def tab_pca(df, band_data_display, bands):
    st.markdown("### " + t("pca_title"))
    st.caption(t("pca_help"))
    if len(df) < 2:
        st.warning(t("warn_min_2_samples"))
        return
    X = band_data_display[bands].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(2, X.shape[0], X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    if X_pca.shape[1] < 2:
        st.warning(t("warn_cannot_2d"))
        return
    labels = [make_row_label_by_pos(df, i) for i in range(len(df))]
    color_col = df["label"].astype(str) if "label" in df.columns else (df["target"].astype(str) if "target" in df.columns else None)
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=color_col,
        hover_name=labels,
        title=t("pca_2d_plot"),
        labels={"x": "PC1", "y": "PC2", "color": t("group")},
    )
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True, key="tab_pca")
    st.caption(f"{t('explained_ratio')}: PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%}")


# =============================================================================
# タブ：Clustering
# =============================================================================
def tab_clustering(df, band_data_display, bands):
    st.markdown("### " + t("clustering_title"))
    st.caption(t("clustering_help"))
    if len(df) < 2:
        st.warning(t("warn_min_2_samples"))
        return
    X = band_data_display[bands].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_clusters = st.slider(t("n_clusters"), 2, min(8, len(df)), min(3, len(df)), key="tab_cluster_k")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    labels = [make_row_label_by_pos(df, i) for i in range(len(df))]
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=[f"Cluster {c}" for c in clusters],
        hover_name=labels,
        title=t("clustering_result"),
        labels={"x": "PC1", "y": "PC2", "color": t("cluster")},
    )
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True, key="tab_clustering")


# =============================================================================
# タブ：BoxPlot
# =============================================================================
def tab_boxplot(band_data_display, bands):
    st.markdown("### " + t("boxplot_title"))
    st.caption(t("boxplot_help"))
    long = band_data_display[bands].melt(var_name="wavelength_nm", value_name="reflectance")
    fig = px.box(long, x="wavelength_nm", y="reflectance", title=t("box_distribution"))
    fig.update_layout(template="plotly_white", height=450, xaxis_title=t("wavelength_nm_col"), yaxis_title=t("reflectance"))
    st.plotly_chart(fig, use_container_width=True, key="tab_boxplot")


def tab_reference(bands):
    st.markdown("### " + t("reference_title"))

    st.markdown("#### " + t("color_converter_title"))
    st.caption(t("color_converter_desc"))

    system_labels = [t("color_system_hex"), t("color_system_rgb"), t("color_system_pantone"), t("color_system_dic")]
    system_internal = ["hex", "rgb", "pantone", "dic"]
    sys_idx = st.selectbox(t("color_system"), range(4), format_func=lambda i: system_labels[i], key="ref_color_system")
    sys = system_internal[sys_idx]
    code = st.text_input(t("color_code"), value="", key="ref_color_code")

    # Mapping uploader (optional)
    mapping_file = st.file_uploader(t("color_map_upload"), type=["csv"], key="ref_color_map")
    mapping_df = None
    if mapping_file is not None:
        try:
            mapping_df = pd.read_csv(mapping_file)
        except Exception:
            mapping_df = None

    template = pd.DataFrame(
        [
            {"system": "PANTONE", "code": "PANTONE 300 C", "r": 0, "g": 94, "b": 184},
            {"system": "DIC", "code": "DIC 255", "r": 230, "g": 0, "b": 18},
        ]
    )
    st.download_button(
        t("color_map_template"),
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="color_mapping_template.csv",
        mime="text/csv",
        use_container_width=True,
        key="ref_color_map_template_dl",
    )

    rgb = None
    if sys == "hex":
        rgb = parse_hex_color(code)
        if rgb is None and code.strip():
            st.warning(t("color_parse_error"))
    elif sys == "rgb":
        rgb = parse_rgb_triplet(code)
        if rgb is None and code.strip():
            st.warning(t("color_parse_error"))
    else:
        if mapping_df is None:
            st.info(t("color_map_needed"))
        else:
            req = {"system", "code", "r", "g", "b"}
            cols = {c.strip() for c in mapping_df.columns}
            if not req.issubset(cols):
                st.warning(t("color_parse_error"))
            else:
                sys_name = "PANTONE" if sys == "pantone" else "DIC"
                m = mapping_df.copy()
                m["system"] = m["system"].astype(str).str.strip().str.upper()
                m["code"] = m["code"].astype(str).str.strip()
                hit = m[(m["system"] == sys_name) & (m["code"].str.lower() == code.strip().lower())]
                if hit.empty:
                    if code.strip():
                        st.warning(t("color_not_found"))
                else:
                    row = hit.iloc[0]
                    try:
                        rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
                    except Exception:
                        rgb = None
                        st.warning(t("color_parse_error"))

    if rgb is not None:
        st.markdown("##### " + t("color_preview"))
        render_color_preview(rgb)

    st.markdown("---")
    st.markdown("#### " + t("wavelength_guide_title"))
    st.caption(t("wavelength_guide_desc"))

    nm_list = [int(b) for b in bands]
    rows = []
    for nm in nm_list:
        desc = wavelength_desc_nm(nm)
        rgb_nm = wavelength_to_rgb(nm)
        rows.append(
            {
                t("wavelength_nm"): nm,
                t("color_hex"): rgb_to_hex(rgb_nm),
                t("color_rgb"): rgb_nm,
                "note": desc,
            }
        )
    df_ref = pd.DataFrame(rows)
    st.dataframe(df_ref, use_container_width=True, hide_index=True)

    st.markdown("#### " + t("color_guide_title"))
    st.caption(t("color_guide_desc"))
    cols = st.columns(6)
    for i, nm in enumerate(nm_list):
        rgb_nm = wavelength_to_rgb(nm)
        hx = rgb_to_hex(rgb_nm)
        with cols[i % 6]:
            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb;border-radius:10px;padding:10px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.04);margin-bottom:10px;">
                  <div style="font-weight:600;font-size:0.9rem;margin-bottom:6px;">{nm} nm</div>
                  <div style="width:100%;height:18px;border-radius:6px;border:1px solid #e5e7eb;background:{hx};"></div>
                  <div style="font-size:0.75rem;color:#6b7280;margin-top:6px;">{hx}<br/>{rgb_nm}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# タブ：Export
# =============================================================================
def tab_export(df, band_data, bands, index_map):
    st.markdown("### " + t("export_title"))
    full_df = st.session_state["df"].iloc[index_map] if index_map else df
    out = full_df.copy()
    st.caption(t("export_caption"))
    csv = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(t("btn_download_csv"), data=csv, file_name="irodori_export.csv", mime="text/csv", use_container_width=True, key="export_btn")


# =============================================================================
# メイン
# =============================================================================
def main():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ja"
    st.set_page_config(page_title=t("page_title"), layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.title("ANSWER Lite")
    st.markdown(t("app_subtitle"))

    result = render_sidebar()
    if result[0] is None:
        st.info(t("info_no_data"))
        return

    df, band_data, bands, index_map = result

    norm_type, smooth_window, display_unit = get_viz_params()
    band_data_display = apply_display_transform(band_data, bands, norm_type, smooth_window, display_unit)

    render_kpis(df, band_data, bands, index_map, st.session_state.get("data_source", ""))

    tab_spectrum_label = t("tab_spectrum")
    tab_compare_label = t("tab_compare")
    tab_diff_label = t("diff_tab")
    tab_ref_label = t("tab_reference")
    tab_pca_label = t("tab_pca")
    tab_cluster_label = t("tab_clustering")
    tab_box_label = t("tab_boxplot")
    tab_export_label = t("tab_export")
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([tab_spectrum_label, tab_compare_label, tab_diff_label, tab_ref_label, tab_pca_label, tab_cluster_label, tab_box_label, tab_export_label])

    with t1:
        tab_spectrum(df, band_data, band_data_display, bands, index_map)
    with t2:
        tab_compare(df, band_data, band_data_display, bands, index_map)
    with t3:
        tab_difference(df, band_data, band_data_display, bands)
    with t4:
        tab_reference(bands)
    with t5:
        tab_pca(df, band_data_display, bands)
    with t6:
        tab_clustering(df, band_data_display, bands)
    with t7:
        tab_boxplot(band_data_display, bands)
    with t8:
        tab_export(df, band_data, bands, index_map)


if __name__ == "__main__":
    main()
