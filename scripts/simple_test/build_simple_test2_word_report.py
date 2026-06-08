#!/usr/bin/env python3
"""Build a docx report for EnAR/outputs/simple_test_2/run_small_dataset."""

from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs/simple_test_2/run_small_dataset"
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"
DEFAULT_DOCX = "反事实小样本分布总结_simple_test_2.docx"
DEFAULT_FIG_DIR = "fig/simple_test_2_report_fig"

FIGURE_ITEMS = [
    ("Origin", "pipeline/envision/original.png"),
    ("Visual impression", "pipeline/envision/impression.png"),
    ("Uncertainty map", "pipeline/envision/uncertainty_heatmap.png"),
    ("Visual impression atten map", "pipeline/attend/counterfactual_attention_heatmap.png"),
    ("Origin atten map", "pipeline/attend/original_attention_heatmap.png"),
    ("单色 pad", "pipeline/attend/mask_origin_overlay.png"),
    ("三色 pad", "pipeline/attend/mask_origin_three_color_overlay.png"),
]

FIGURE_SLUGS = {
    "Origin": "origin",
    "Visual impression": "visual_impression",
    "Uncertainty map": "uncertainty_map",
    "Visual impression atten map": "visual_impression_atten_map",
    "Origin atten map": "origin_atten_map",
    "单色 pad": "mono_pad",
    "三色 pad": "three_color_pad",
}

PREFERRED_SAMPLES = {
    "Animals": [
        "animal_053_arcticfox_notitle_Q2_px768",
        "animal_030_lynx_notitle_Q1_px1152",
    ],
    "Chess Pieces": [
        "chess_pieces_004_replace_notitle_px768_prompt2",
        "xiangqi_pieces_003_remove_notitle_px1152_prompt1",
    ],
    "Flags": [
        "Flag_of_Malaysia_stripes_13_768_1",
        "flag_stars_024_notitle_Q2_px1152",
    ],
    "Game Boards": [
        "chess_grid_07_col_add_first_col_notitle_px1152_Q2",
        "chess_grid_04_row_add_last_row_notitle_px1152_Q2",
        "chess_grid_08_col_add_last_col_notitle_px1152_Q1",
    ],
    "Logos": [
        "car_065_notitle_Q1_px1152",
        "car_066_notitle_Q2_px768",
    ],
    "Optical Illusion": [
        "Zollner_008_Q1_notitle_px1152",
        "Poggendorff_016_Q1_notitle_px1152",
    ],
    "Patterned Grid": [
        "dice_007_remove_notitle_px384_prompt1",
        "tally_012_remove_notitle_px768_prompt1",
    ],
}

TOPIC_COMMENTS = {
    "Animals": "EnAR 更容易把答案推回常识腿数 4，expected bias rate 从 60.00% 升到 70.00%。",
    "Chess Pieces": "棋子计数整体失败，Regular 常给 15/155/185 一类文本答案，EnAR 常变成 18 或坐标片段。",
    "Flags": "条纹计数相对稳定，星星计数仍向 13/15/5 等固定值偏移；EnAR 新增了一个 bias hit。",
    "Game Boards": "本轮最明显的正迁移 topic，EnAR accuracy 从 10.00% 升到 20.00%，bias rate 从 40.00% 降到 20.00%。",
    "Logos": "EnAR 打断了部分品牌先验 bias，但几乎统一输出 13，因此 bias 下降没有转化成正确率。",
    "Optical Illusion": "Yes/No 型样本最稳定，Regular 与 EnAR 输出基本一致，保持 40.00% accuracy。",
    "Patterned Grid": "局部格子计数仍不稳定，EnAR 常将 100/15 改成 10/13，但没有对齐 ground truth。",
}


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    docs_dir = args.docs_dir.expanduser().resolve()
    fig_dir = resolve_fig_dir(args.fig_dir, docs_dir)
    docx_path = docs_dir / args.output_name

    records = read_jsonl(run_dir / "predictions.jsonl")
    metrics = read_json(run_dir / "metrics.json")
    selected = select_samples(records)
    fig_dir.mkdir(parents=True, exist_ok=True)

    builder = DocxBuilder()
    build_report(builder, records, metrics, selected, fig_dir)
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    builder.save(docx_path)

    print(f"DOCX: {docx_path}")
    print(f"FIG:  {fig_dir}")
    print(f"Samples: {sum(len(v) for v in selected.values())}")
    print(f"Images: {len(builder.media)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simple_test_2 counterfactual Word report.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--fig-dir", type=Path, default=Path(DEFAULT_FIG_DIR))
    parser.add_argument("--output-name", default=DEFAULT_DOCX)
    return parser.parse_args()


def build_report(
    doc: "DocxBuilder",
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    selected: dict[str, list[dict[str, Any]]],
    fig_dir: Path,
) -> None:
    doc.heading("反事实小样本分布规律总结（simple_test_2）", level=1, center=True)
    doc.paragraph("参考结构：复现.docx；本报告基于 EnAR/outputs/simple_test_2/run_small_dataset。", italic=True)
    regular = metrics["regular"]
    enar = metrics["enar"]
    delta = metrics["delta"]
    doc.paragraph(
        f"本轮共评估 {metrics['num_evaluated']} 个反事实样本。Regular accuracy 为 "
        f"{pct(regular['overall_accuracy'])}（{regular['correct_count']}/{regular['evaluated_count']}），"
        f"EnAR accuracy 为 {pct(enar['overall_accuracy'])}（{enar['correct_count']}/{enar['evaluated_count']}），"
        f"整体提升 {signed_pct(delta['overall_accuracy'])}。Expected bias rate 从 "
        f"{pct(regular['expected_bias_rate'])} 降到 {pct(enar['expected_bias_rate'])}。"
    )
    doc.paragraph(
        "核心现象：整体提升很小，但 Game Boards 有正迁移；Animals 更偏向常识腿数；"
        "Logos 的 bias hit 下降但出现 13 的输出塌缩；Optical Illusion 基本不受干预影响。"
    )

    doc.heading("1. 结果分布", level=2)
    doc.table([
        ["Method", "Accuracy", "Correct / Evaluated", "Expected Bias Rate", "Expected Bias Hits"],
        ["Regular", pct(regular["overall_accuracy"]), f"{regular['correct_count']} / {regular['evaluated_count']}", pct(regular["expected_bias_rate"]), str(regular["expected_bias_hits"])],
        ["EnAR", pct(enar["overall_accuracy"]), f"{enar['correct_count']} / {enar['evaluated_count']}", pct(enar["expected_bias_rate"]), str(enar["expected_bias_hits"])],
        ["Delta", signed_pct(delta["overall_accuracy"]), "", signed_pct(delta["expected_bias_rate"]), ""],
    ])

    doc.heading("按 Topic 的准确率与 Bias 命中率", level=3)
    rows = [["Topic", "Count", "Regular Acc", "EnAR Acc", "Delta Acc", "Regular Bias", "EnAR Bias", "Delta Bias"]]
    for topic, row in metrics["by_topic"].items():
        rows.append([
            topic,
            str(row["count"]),
            pct(row["regular_accuracy"]),
            pct(row["enar_accuracy"]),
            signed_pct(row["delta_accuracy"]),
            pct(row["regular_expected_bias_rate"]),
            pct(row["enar_expected_bias_rate"]),
            signed_pct(row["delta_expected_bias_rate"]),
        ])
    doc.table(rows)

    doc.heading("Outcome 分布", level=3)
    outcomes = metrics.get("outcome_distribution", {})
    doc.table([
        ["Outcome", "Count"],
        ["Regular/EnAR 均错误", str(outcomes.get("both_wrong", 0))],
        ["二者均命中 expected bias", str(outcomes.get("both_hit_expected_bias", 0))],
        ["二者均正确", str(outcomes.get("both_correct", 0))],
        ["仅 EnAR 正确", str(outcomes.get("enar_only_correct", 0))],
        ["仅 Regular 正确", str(outcomes.get("regular_only_correct", 0))],
    ])

    doc.heading("答案分布观察", level=3)
    reg_dist = metrics.get("answer_distribution", {}).get("regular", {}).get("by_topic", {})
    enar_dist = metrics.get("answer_distribution", {}).get("enar", {}).get("by_topic", {})
    rows = [["Topic", "Regular top answers", "EnAR top answers"]]
    for topic in metrics["by_topic"]:
        rows.append([topic, top_answers(reg_dist.get(topic, {})), top_answers(enar_dist.get(topic, {}))])
    doc.table(rows)

    doc.heading("2. 分布规律总结", level=2)
    for item in [
        "整体：EnAR 从 6/70 提升到 7/70，提升来自 Game Boards 的两个 EnAR-only correct；但 62 个样本仍然二者均错。",
        "Bias：整体 expected bias rate 小幅下降，但 topic 内方向不一致。Animals 与 Flags 上升，Game Boards 与 Logos 下降。",
        "输出塌缩：计数类问题中 EnAR 高频输出 13、18、9、4，部分样本输出坐标格式片段，说明 Respond 阶段仍有答案格式不稳定。",
        "Game Boards：两个 chess grid 样本从 regular 错误的 13 被 EnAR 修正为 9，是本轮最典型正例；另一个 chess grid 样本则从 regular 正确变成 EnAR 错误。",
        "Logos：Regular 有少量品牌常识偏置，EnAR 消除了这些 bias hit，但几乎全部回答 13，说明注意力/pad 干预未定位到真实计数对象。",
        "Pad 图解释：单色 pad 是最终 union padding 区域；三色 pad 中红色为 attention-only、蓝色为 uncertainty-only、黄色为二者重合。",
    ]:
        doc.bullet(item)

    doc.heading("3. Topic 经典 Sample 可视化", level=2)
    doc.paragraph(
        "每个 sample 包含七类图：origin、visual impression、uncertainty map、"
        "visual impression 的 attention map、origin 的 attention map、单色 pad、三色 pad。"
    )
    for topic, topic_records in selected.items():
        doc.page_break()
        doc.heading(topic, level=2)
        doc.paragraph(TOPIC_COMMENTS.get(topic, ""))
        for record in topic_records:
            add_sample(doc, record, fig_dir)


def add_sample(doc: "DocxBuilder", record: dict[str, Any], fig_dir: Path) -> None:
    sample_id = record["sample_id"]
    base = Path(record["paths"]["sample_json"]).parent
    doc.heading(sample_id, level=3)
    doc.table([
        ["Question", record.get("prompt", "")],
        ["GT / Expected bias", f"{record.get('ground_truth')} / {record.get('expected_bias')}"],
        ["Regular", f"{record['regular']['answer']} | correct={record['regular']['correct']} | bias={record['regular']['hits_expected_bias']}"],
        ["EnAR", f"{record['enar']['answer']} | correct={record['enar']['correct']} | bias={record['enar']['hits_expected_bias']}"],
        ["Why selected", sample_reason(record)],
    ])
    figure_rows = []
    for chunk in chunks(FIGURE_ITEMS, 4):
        row = []
        for title, rel_path in chunk:
            src = base / rel_path
            dst = fig_dir / f"{safe_name(sample_id)}__{FIGURE_SLUGS[title]}.png"
            prepare_image(src, dst)
            row.append((title, dst))
        figure_rows.append(row)
    doc.image_grid(figure_rows)


def select_samples(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id = {record["sample_id"]: record for record in records}
    output = {}
    for topic in sorted({record["topic"] for record in records}):
        chosen = [by_id[item] for item in PREFERRED_SAMPLES.get(topic, []) if item in by_id]
        if len(chosen) < 2:
            pool = [record for record in records if record["topic"] == topic and record not in chosen]
            pool.sort(key=interesting_score, reverse=True)
            chosen.extend(pool[: 2 - len(chosen)])
        output[topic] = chosen
    return output


def interesting_score(record: dict[str, Any]) -> int:
    reg = record["regular"]
    enar = record["enar"]
    score = 0
    if reg["correct"] != enar["correct"]:
        score += 50
    if reg["hits_expected_bias"] != enar["hits_expected_bias"]:
        score += 30
    if reg["hits_expected_bias"] and enar["hits_expected_bias"]:
        score += 20
    if reg["answer"] != enar["answer"]:
        score += 10
    return score


def sample_reason(record: dict[str, Any]) -> str:
    reg = record["regular"]
    enar = record["enar"]
    if reg["correct"] and enar["correct"]:
        return "Regular 与 EnAR 均正确，用作正例参照。"
    if not reg["correct"] and enar["correct"]:
        return "EnAR-only correct，本轮正迁移代表样本。"
    if reg["correct"] and not enar["correct"]:
        return "Regular-only correct，说明 EnAR 干预破坏了原本正确输出。"
    if reg["hits_expected_bias"] and enar["hits_expected_bias"]:
        return "二者均命中 expected bias，是稳定偏置失败案例。"
    if reg["hits_expected_bias"] and not enar["hits_expected_bias"]:
        return "EnAR 打断 expected bias，但未转化为正确答案。"
    if not reg["hits_expected_bias"] and enar["hits_expected_bias"]:
        return "EnAR 将输出推向 expected bias，是负向迁移案例。"
    return "二者均错误但不命中 expected bias，用于观察输出塌缩和 pad 区域。"


def prepare_image(src: Path, dst: Path, max_size: tuple[int, int] = (520, 420)) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        image = Image.new("RGB", max_size, (245, 245, 245))
    else:
        image = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", max_size, "white")
    canvas.paste(image, ((max_size[0] - image.width) // 2, (max_size[1] - image.height) // 2))
    canvas.save(dst, quality=92)


class DocxBuilder:
    def __init__(self) -> None:
        self.blocks: list[str] = []
        self.rels: list[tuple[str, str, str]] = []
        self.media: list[tuple[Path, str]] = []
        self.next_rel = 1
        self.next_pic = 1

    def heading(self, text: str, *, level: int = 1, center: bool = False) -> None:
        self.blocks.append(self.p(text, bold=True, size={1: 32, 2: 26, 3: 22}.get(level, 20), center=center))

    def paragraph(self, text: str, *, italic: bool = False) -> None:
        self.blocks.append(self.p(text, italic=italic))

    def bullet(self, text: str) -> None:
        self.blocks.append(self.p("• " + text, indent=360))

    def page_break(self) -> None:
        self.blocks.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def table(self, rows: list[list[str]]) -> None:
        table_rows = []
        for i, row in enumerate(rows):
            cells = []
            for cell in row:
                fill = '<w:shd w:fill="E9EEF5"/>' if i == 0 else ""
                cells.append("<w:tc><w:tcPr>" + fill + "</w:tcPr>" + self.p(str(cell), bold=i == 0, size=18) + "</w:tc>")
            table_rows.append("<w:tr>" + "".join(cells) + "</w:tr>")
        self.blocks.append(
            "<w:tbl><w:tblPr><w:tblBorders>"
            '<w:top w:val="single" w:sz="4" w:color="B8C0CC"/>'
            '<w:left w:val="single" w:sz="4" w:color="B8C0CC"/>'
            '<w:bottom w:val="single" w:sz="4" w:color="B8C0CC"/>'
            '<w:right w:val="single" w:sz="4" w:color="B8C0CC"/>'
            '<w:insideH w:val="single" w:sz="4" w:color="B8C0CC"/>'
            '<w:insideV w:val="single" w:sz="4" w:color="B8C0CC"/>'
            "</w:tblBorders></w:tblPr>"
            + "".join(table_rows)
            + "</w:tbl>"
        )

    def image_grid(self, rows: list[list[tuple[str, Path]]]) -> None:
        table_rows = []
        for row in rows:
            cells = []
            for title, path in row:
                cells.append("<w:tc>" + self.p(title, bold=True, size=17, center=True) + self.image(path, width_inches=1.45) + "</w:tc>")
            table_rows.append("<w:tr>" + "".join(cells) + "</w:tr>")
        self.blocks.append(
            "<w:tbl><w:tblPr><w:tblBorders>"
            '<w:top w:val="single" w:sz="4" w:color="D0D7DE"/>'
            '<w:left w:val="single" w:sz="4" w:color="D0D7DE"/>'
            '<w:bottom w:val="single" w:sz="4" w:color="D0D7DE"/>'
            '<w:right w:val="single" w:sz="4" w:color="D0D7DE"/>'
            '<w:insideH w:val="single" w:sz="4" w:color="D0D7DE"/>'
            '<w:insideV w:val="single" w:sz="4" w:color="D0D7DE"/>'
            "</w:tblBorders></w:tblPr>"
            + "".join(table_rows)
            + "</w:tbl>"
        )

    def p(
        self,
        text: str,
        *,
        bold: bool = False,
        italic: bool = False,
        size: int = 21,
        center: bool = False,
        indent: int = 0,
    ) -> str:
        jc = '<w:jc w:val="center"/>' if center else ""
        ind = f'<w:ind w:left="{indent}"/>' if indent else ""
        rpr = (
            '<w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/>'
            + ('<w:b/>' if bold else '')
            + ('<w:i/>' if italic else '')
            + f'<w:sz w:val="{size}"/></w:rPr>'
        )
        return f'<w:p><w:pPr>{jc}{ind}<w:spacing w:after="90"/></w:pPr><w:r>{rpr}<w:t xml:space="preserve">{xml(text)}</w:t></w:r></w:p>'

    def image(self, path: Path, *, width_inches: float) -> str:
        rel_id, pic_id = self.add_image(path)
        with Image.open(path) as image:
            width_px, height_px = image.size
        width = int(width_inches * 914400)
        height = int(width * height_px / max(width_px, 1))
        drawing = f"""
<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{width}" cy="{height}"/><wp:docPr id="{pic_id}" name="Picture {pic_id}"/>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic><pic:nvPicPr><pic:cNvPr id="{pic_id}" name="{xml(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="{height}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>
"""
        return f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r>{drawing}</w:r></w:p>'

    def add_image(self, path: Path) -> tuple[str, int]:
        rel_id = f"rId{self.next_rel}"
        self.next_rel += 1
        pic_id = self.next_pic
        self.next_pic += 1
        target = f"media/image{pic_id}.png"
        self.rels.append((rel_id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", target))
        self.media.append((path, target))
        return rel_id, pic_id

    def save(self, path: Path) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", CONTENT_TYPES)
            zf.writestr("_rels/.rels", ROOT_RELS)
            zf.writestr("word/document.xml", self.document_xml())
            zf.writestr("word/_rels/document.xml.rels", self.rels_xml())
            zf.writestr("docProps/core.xml", CORE_XML)
            zf.writestr("docProps/app.xml", APP_XML)
            for src, target in self.media:
                zf.write(src, "word/" + target)

    def document_xml(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<w:body>{''.join(self.blocks)}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="850" w:right="850" w:bottom="850" w:left="850"/></w:sectPr></w:body></w:document>"""

    def rels_xml(self) -> str:
        rels = "".join(f'<Relationship Id="{rid}" Type="{typ}" Target="{target}"/>' for rid, typ, target in self.rels)
        return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>'


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

CORE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>反事实小样本分布规律总结 simple_test_2</dc:title><dc:creator>Codex</dc:creator></cp:coreProperties>"""

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex OOXML Writer</Application></Properties>"""


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_fig_dir(path: Path, docs_dir: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (docs_dir / path).resolve()


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def top_answers(counts: dict[str, int], limit: int = 3) -> str:
    return "; ".join(f"{answer or '<blank>'} ({count})" for answer, count in Counter(counts).most_common(limit))


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")


def xml(value: str) -> str:
    return html.escape(str(value), quote=False)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def signed_pct(value: float) -> str:
    return f"{100 * value:+.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
