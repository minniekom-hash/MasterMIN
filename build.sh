#!/usr/bin/env sh
# Render build script.
#
# Kept deliberately boring so a deploy can never die on an optional extra:
# every step that is not strictly required to boot the app is allowed to fail.
set -e

echo "==> installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# Optional: VADER sentiment. omega/news.py falls back to a built-in finance
# lexicon when nltk is missing, so a failure here must not break the build.
echo "==> optional sentiment model (safe to skip)"
pip install nltk >/dev/null 2>&1 || true
python - <<'PY' || true
try:
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    print("sentiment: NLTK VADER ready")
except Exception as ex:
    print("sentiment: skipping (%s) -> built-in lexicon will be used" % type(ex).__name__)
PY

echo "==> verifying the app imports"
python -c "import app; print('app import OK')"
echo "==> build finished"
