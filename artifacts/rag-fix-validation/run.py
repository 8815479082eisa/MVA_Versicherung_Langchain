import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.run_pdf_rag_200 import collect_case

base = Path(__file__).resolve().parent
cases = [json.loads(x) for x in (ROOT / 'artifacts/hallucination-100-current-2026-09-11/dataset_100.jsonl').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
if len(sys.argv) > 1:
    cases = [c for c in cases if c['id'] in sys.argv[1:]]
path = base / ('smoke.jsonl' if len(cases) < 100 else 'responses.jsonl')
results = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() and len(cases) == 100 else []
completed = {r['id'] for r in results}
with path.open('a' if results else 'w', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(collect_case, c, 'http://127.0.0.1:8001/api/ask', 240) for c in cases if c['id'] not in completed]
    for future in as_completed(futures):
        row = future.result()
        results.append(row)
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        print(len(results), row['id'], row['status_code'], str(row['payload'].get('answer', ''))[:100], flush=True)
print(dict(Counter(r['status_code'] for r in results)), flush=True)
