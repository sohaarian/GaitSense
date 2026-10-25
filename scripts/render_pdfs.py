"""Render the technical report and faculty summary to PDF (pandoc + wkhtmltopdf)."""
import re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOCS = [ROOT / "docs" / "technical_report.md", ROOT / "docs" / "faculty" / "faculty_summary.md"]


def fix_lists(text):
    """Pandoc needs a blank line before a list that follows a paragraph line."""
    out = []
    for line in text.splitlines():
        if re.match(r"\s*(?:>\s*)?(?:[-*]|\d+\.)\s", line) and out and out[-1].strip() not in ("", ">") \
                and not re.match(r"\s*(?:>\s*)?(?:[-*]|\d+\.)\s", out[-1]):
            out.append(">" if line.lstrip().startswith(">") else "")
        out.append(line)
    return "\n".join(out) + "\n"


for md in (sys.argv[1:] and [Path(a) for a in sys.argv[1:]]) or DOCS:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / md.name; src.write_text(fix_lists(md.read_text()))
        html = Path(tmp) / "out.html"
        subprocess.run(["pandoc", str(src), "-s", "--embed-resources", "--css", str(ROOT / "docs" / "pdf.css"),
                        "--resource-path", str(md.parent), "-o", str(html)], check=True)
        subprocess.run(["wkhtmltopdf", "--quiet", "--enable-local-file-access", "-s", "Letter",
                        "-T", "20mm", "-B", "20mm", "-L", "22mm", "-R", "22mm",
                        str(html), str(md.with_suffix(".pdf"))], check=True)
    print("wrote", md.with_suffix(".pdf"))
