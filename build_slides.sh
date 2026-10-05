#!/usr/bin/env bash
# Build Bank of England-styled slides from a notebook: <name>.slides.html and <name>.slides.pdf
#
#   ./build_slides.sh                      # main_ccbs.ipynb
#   ./build_slides.sh other.ipynb          # another notebook
#   EXECUTE=1 ./build_slides.sh            # re-run the notebook first (regenerates the figures)
#
# Needs: jupyter nbconvert, Google Chrome/Chromium, and internet access (reveal.js + MathJax load from CDNs).
set -euo pipefail
cd "$(dirname "$0")"

NB="${1:-main_ccbs.ipynb}"
NAME="${NB%.ipynb}"
HTML="$NAME.slides.html"
PDF="$NAME.slides.pdf"

if [[ "${EXECUTE:-0}" == "1" ]]; then
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$NB"
fi

jupyter nbconvert "$NB" --to slides \
  --TemplateExporter.extra_template_basedirs=slides_template \
  --template=boe \
  --HTMLExporter.embed_images=True \
  --output "$NAME"

CHROME="${CHROME:-$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)}"
if [[ -z "$CHROME" ]]; then
  echo "Chrome/Chromium not found: wrote $HTML only (open it with ?print-pdf and print to PDF manually)." >&2
  exit 0
fi

"$CHROME" --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
  --allow-file-access-from-files \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=60000 \
  --no-pdf-header-footer \
  --print-to-pdf="$PWD/$PDF" \
  "file://$PWD/$HTML?print-pdf" 2>/dev/null

echo "Wrote $HTML and $PDF ($(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages/{print $2}') pages)"
