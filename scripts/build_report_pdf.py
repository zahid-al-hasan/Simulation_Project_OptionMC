"""Build the final Markdown report as a styled PDF with local figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import markdown


CSS = """
@page { size: A4; margin: 16mm 15mm 17mm; }
body {
  font-family: Arial, Helvetica, sans-serif;
  color: #172033;
  line-height: 1.48;
  font-size: 10.2pt;
}
h1 { color: #153b66; font-size: 25pt; margin: 0 0 16px; }
h2 { color: #153b66; font-size: 17pt; margin-top: 24px; break-after: avoid; }
h3 { color: #24557f; font-size: 13pt; margin-top: 20px; break-after: avoid; }
p { orphans: 3; widows: 3; }
img {
  display: block;
  max-width: 100%;
  max-height: 225mm;
  width: auto;
  height: auto;
  margin: 10px auto 18px;
  break-inside: avoid;
  page-break-inside: avoid;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 7.4pt;
  margin: 10px 0 16px;
  break-inside: avoid;
}
th { background: #e8f0f7; color: #153b66; }
th, td { border: 1px solid #9aaabd; padding: 4px 5px; text-align: right; }
th:first-child, th:nth-child(2), td:first-child, td:nth-child(2) { text-align: left; }
code {
  font-family: Consolas, monospace;
  background: #f2f4f7;
  padding: 1px 3px;
  border-radius: 3px;
}
pre {
  background: #f2f4f7;
  border-left: 4px solid #4c83b6;
  padding: 10px;
  white-space: pre-wrap;
  break-inside: avoid;
}
li { margin-bottom: 4px; }
a { color: #1c5d99; text-decoration: none; }
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("FINAL_PROJECT_REPORT.md"))
    parser.add_argument("--output", type=Path, default=Path("FINAL_PROJECT_REPORT.pdf"))
    return parser.parse_args()


def find_chrome() -> Path:
    candidates = (
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    executable = shutil.which("chrome") or shutil.which("msedge")
    if executable:
        return Path(executable)
    raise FileNotFoundError("Chrome or Edge is required to render the report PDF")


def main():
    args = parse_args()
    project_root = Path.cwd().resolve()
    source = args.input.resolve()
    output = args.output.resolve()
    temp_dir = project_root / "tmp" / "pdfs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    html_path = temp_dir / "FINAL_PROJECT_REPORT.html"
    temporary_pdf = temp_dir / "FINAL_PROJECT_REPORT.rendered.pdf"

    markdown_text = source.read_text(encoding="utf-8")
    body = markdown.markdown(markdown_text, extensions=("tables", "fenced_code"))
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<base href=\"{project_root.as_uri()}/\">"
        "<title>Variance Reduction Techniques for Monte Carlo Option Pricing</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")

    chrome = find_chrome()
    profile_dir = temp_dir / "chrome-profile"
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile_dir}",
        f"--print-to-pdf={temporary_pdf}",
        html_path.as_uri(),
    ]
    subprocess.run(command, check=True)
    if not temporary_pdf.exists() or temporary_pdf.stat().st_size == 0:
        raise RuntimeError("browser completed without creating a PDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf.replace(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
