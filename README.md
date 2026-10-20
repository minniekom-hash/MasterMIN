# OPTION TRADE: Omega Singularity Engine

ระบบวิเคราะห์ออปชันรายสัปดาห์ด้วยโมเดล 5 ชั้น (Monte Carlo, BSM Greeks, Order Flow, Technical, NLP) — เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน

## รันในเครื่อง
    pip install -r requirements.txt
    python app.py   # เปิด http://localhost:5000

## Deploy บน Render.com
1. Push โค้ดขึ้น GitHub
2. Render → New → Blueprint (ใช้ `render.yaml`) หรือ New Web Service
   - Build: `pip install -r requirements.txt && python -c "import nltk; nltk.download('vader_lexicon')"`
   - Start: `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
3. ตั้ง Environment: `NEWSAPI_KEY` (ไม่บังคับ), `WATCHLIST`, `RISK_FREE`

Railway: ใช้ `Procfile` ได้ทันที
หมายเหตุ: ต้องใช้ `--workers 1` เพราะตัวสแกนทำงานเป็น background thread
