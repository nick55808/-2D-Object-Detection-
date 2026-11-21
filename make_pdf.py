#!/usr/bin/env python3
"""Convert the generated Markdown report to a polished PDF.

Pandoc is preferred when a TeX format is available. The managed runtime may
ship the TeX binaries without prebuilt format files, so a ReportLab fallback is
included to keep PDF generation reproducible in that environment.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    markdown = Path(args.markdown).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Use Pandoc only when XeLaTeX has a usable format file. Otherwise use the
    # deterministic ReportLab fallback below.
    if shutil.which("pandoc") and shutil.which("xelatex"):
        probe = subprocess.run(["xelatex", "--version"], capture_output=True, text=True)
        if probe.returncode == 0:
            command = [
                "pandoc", str(markdown), "--from", "markdown", "--standalone",
                "--number-sections", "--pdf-engine=xelatex",
                "-V", "geometry:margin=0.72in",
                "-V", "mainfont:DejaVu Sans",
                "-V", "monofont:DejaVu Sans Mono",
                "-V", "colorlinks:true",
                "--resource-path", str(markdown.parent), "-o", str(output),
            ]
            try:
                print("$", " ".join(command))
                subprocess.run(command, check=True)
                print(f"Wrote {output}")
                return
            except subprocess.CalledProcessError:
                print("Pandoc/XeLaTeX failed; using the ReportLab fallback.")

    build_reportlab_pdf(markdown, output)
    print(f"Wrote {output}")


def inline_markup(text: str) -> str:
    """Convert the small Markdown subset used by the report into Paragraph XML."""
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\mathrm", "")
    text = escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


def parse_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for raw in lines:
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if cells and all(set(cell.replace("-", "").replace(":", "").strip()) == set() for cell in cells):
            continue
        if cells:
            rows.append([Paragraph(inline_markup(cell), styles["table"]) for cell in cells])
    column_count = max(len(row) for row in rows)
    for row in rows:
        row.extend([Paragraph("", styles["table"])] * (column_count - len(row)))
    width = 7.05 * inch
    table_obj = Table(rows, colWidths=[width / column_count] * column_count, repeatRows=1)
    table_obj.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b5b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b7c3d0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]),
    ]))
    return table_obj


def build_reportlab_pdf(markdown: Path, output: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, alignment=TA_CENTER, textColor=colors.HexColor("#17324d"), spaceAfter=14))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#4b6478"), spaceAfter=13))
    styles.add(ParagraphStyle(name="H1Report", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#1f4e79"), spaceBefore=12, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name="H2Report", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=colors.HexColor("#2d6a9f"), spaceBefore=8, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="H3Report", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.HexColor("#2d6a9f"), spaceBefore=6, spaceAfter=3, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyReport", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.9, leading=11.7, spaceAfter=5, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="BulletReport", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.3, leading=10.2, leftIndent=14, firstLineIndent=-8, spaceAfter=1))
    styles.add(ParagraphStyle(name="CodeReport", fontName="Courier", fontSize=6.8, leading=8.2, leftIndent=8, backColor=colors.HexColor("#f4f6f8"), borderPadding=5, spaceBefore=3, spaceAfter=6))
    styles.add(ParagraphStyle(name="table", parent=styles["BodyText"], fontName="Helvetica", fontSize=5.5, leading=6.6, spaceAfter=0))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.5, leading=9, textColor=colors.HexColor("#5b6770"), spaceAfter=5))

    raw_lines = markdown.read_text(encoding="utf-8").splitlines()
    # Remove YAML front matter.
    if raw_lines and raw_lines[0].strip() == "---":
        try:
            end = raw_lines.index("---", 1)
            raw_lines = raw_lines[end + 1:]
        except ValueError:
            pass

    flowables = []
    i = 0
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            text = " ".join(part.strip() for part in paragraph_buffer).strip()
            if text:
                flowables.append(Paragraph(inline_markup(text), styles["BodyReport"]))
            paragraph_buffer.clear()

    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.strip()
        if not stripped or stripped == "---":
            flush_paragraph()
            i += 1
            continue
        if stripped.startswith("# Appendix C"):
            flush_paragraph()
            flowables.append(PageBreak())
            flowables.append(Paragraph(inline_markup(stripped[2:]), styles["H1Report"]))
            i += 1
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            flowables.append(Paragraph(inline_markup(stripped[2:]), styles["ReportTitle"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flowables.append(Paragraph(inline_markup(stripped[3:]), styles["H1Report"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            flowables.append(Paragraph(inline_markup(stripped[4:]), styles["H2Report"]))
            i += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            code = []
            i += 1
            while i < len(raw_lines) and not raw_lines[i].strip().startswith("```"):
                code.append(raw_lines[i])
                i += 1
            flowables.append(Preformatted("\n".join(code), styles["CodeReport"]))
            i += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(raw_lines) and raw_lines[i].strip().startswith("|"):
                table_lines.append(raw_lines[i])
                i += 1
            flowables.append(parse_table(table_lines, styles))
            flowables.append(Spacer(1, 5))
            continue
        image_match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            image_path = (markdown.parent / image_match.group(2)).resolve()
            if image_path.exists():
                image = Image(str(image_path))
                image._restrictSize(7.0 * inch, 3.8 * inch)
                flowables.append(image)
                flowables.append(Paragraph(inline_markup(image_match.group(1)), styles["Caption"]))
            else:
                flowables.append(Paragraph(f"[Figure unavailable: {escape(image_match.group(1))}]", styles["Caption"]))
            i += 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            flowables.append(Paragraph("- " + inline_markup(stripped[2:]), styles["BulletReport"]))
            i += 1
            continue
        if re.match(r"^\d+\. ", stripped):
            flush_paragraph()
            flowables.append(Paragraph(inline_markup(stripped), styles["BulletReport"]))
            i += 1
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            flowables.append(Paragraph("<i>" + inline_markup(stripped[2:]) + "</i>", styles["BodyReport"]))
            i += 1
            continue
        paragraph_buffer.append(stripped)
        i += 1
    flush_paragraph()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d7dee5"))
        canvas.line(0.72 * inch, 0.52 * inch, 7.78 * inch, 0.52 * inch)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6b7785"))
        canvas.drawString(0.72 * inch, 0.32 * inch, "Track A - 2D Object Detection Evaluation")
        canvas.drawRightString(7.78 * inch, 0.32 * inch, f"Page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output), pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch,
        topMargin=0.62 * inch, bottomMargin=0.72 * inch,
        title="2D Object Detection Track A Report",
    )
    document.build(flowables, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
