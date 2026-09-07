# گزارش نهایی رفع CRM و Output Guardrail و اجرای مجدد Routing Evaluation

تاریخ اجرا: 2026-08-22  
API: `http://127.0.0.1:8000/api/ask`  
Dataset: `data/benchmarks/routing/routing_eval_80.jsonl`

## خلاصه اجرایی

هر دو خطای عملیاتی اصلی رفع شدند:

- EspoCRM در backend فعال است و ابزارهای واقعی CRM داده برمی‌گردانند.
- مدل ONNX مورد استفاده NeMo Output Guardrail در cache پایدار کامل است و inference واقعی موفق دارد.
- در اجرای مجدد ۸۰ سؤال، هیچ `CRM_REQUEST_FAILED`، هیچ `GUARDRAIL_INVALID_OUTPUT` و هیچ HTTP 503 مشاهده نشد.
- Routing Accuracy برابر 79/80 یا 98.75% باقی ماند.
- Strict Operational End-to-End Success برابر 79/80 یا 98.75% شد.
- Downstream Operational Completion مستقل از routing برابر 80/80 یا 100% بود.

## 1. اصلاح و اثبات CRM

علت خطای قبلی این بود که کانتینر `mva-backend` فقط با compose پایه ساخته شده بود. در نتیجه تنظیمات `docker-compose.crm.yml` روی backend اعمال نشده و `/health` مقدارهای `crmEnabled=false` و `crmReady=false` نشان می‌داد.

backend با فایل‌های compose پایه و CRM و متغیرهای `.env.crm` دوباره ساخته شد. پس از راه‌اندازی:

- `crmEnabled=true`
- `crmReady=true`
- `crmError=null`
- ابزارهای MCP آماده: `find_customer`, `get_policy`, `get_claim_status`, `get_customer_claims`, `get_customer_policies`

اثبات فقط readiness نبود. درخواست واقعی زیر از API اجرا شد:

`What annual premium is recorded for policy TEST-KFZ-2026-1001?`

نتیجه:

- HTTP 200
- route: `crm-only`
- status: `complete`
- CRM stage: `completed`
- policy: `TEST-KFZ-2026-1001`
- customer: `Lara Neumann`
- annual premium: `684 EUR`
- deductible: `150 EUR`
- CRM latency: حدود `153 ms`

## 2. اصلاح و اثبات Output Guardrail

FastEmbed به‌صورت پیش‌فرض cache را در Temp قرار می‌داد. cache ناقص قبلی باعث می‌شد `model.onnx` پیدا نشود. مسیر پایدار زیر برای کانتینر تنظیم شد:

`FASTEMBED_CACHE_PATH=/app/.cache/fastembed`

مدل واقعی FastEmbed:

`sentence-transformers/all-MiniLM-L6-v2`  
Hugging Face repository: `qdrant/all-MiniLM-L6-v2-onnx`

فایل‌های اصلی از داخل کانتینر بررسی شدند:

- `model.onnx`: 90,387,630 bytes
- `tokenizer.json`: 711,661 bytes
- `config.json`: 650 bytes

readiness جدید تنها وجود فایل را بررسی نمی‌کند. مدل را load می‌کند، یک embedding واقعی می‌سازد و بعد dimension مورد انتظار 384 را کنترل می‌کند. health نهایی:

- `guardrailModelReady=true`
- `guardrailEmbeddingDimension=384`
- `guardrailModelError=null`
- `llmReady=true`

یک سؤال واقعی RAG نیز اجرا شد. پاسخ HTTP 200 و `status=complete` بود و مراحل `guardrail` و `groundedness` هر دو `completed` شدند. تصمیم نهایی safety برابر `allow` بود.

نکته: `ollamaReachable=false` باقی مانده، اما مسیر فعال answer از OpenAI استفاده می‌کند و Output Guardrail فعال در حالت embeddings-only به FastEmbed وابسته است. برای شفافیت، health مقدار اختیاری `ollamaGuardrailModelReady=false` را جداگانه نگه می‌دارد.

## 3. اثبات عدم تغییر Dataset

SHA-256 قبل و بعد از اجرای API کاملاً یکسان بود:

`1A65CF29C6CA11304A3265C7CE21EB38624B8EE81C195A01C1E06C0A63CF91D3`

بنابراین فایل `routing_eval_80.jsonl` بدون تغییر اجرا شده است.

## 4. Routing Accuracy

| مسیر مورد انتظار | صحیح | کل | Recall |
| --- | ---: | ---: | ---: |
| `crm_only` | 20 | 20 | 100% |
| `retrieval_only` | 20 | 20 | 100% |
| `combined` | 20 | 20 | 100% |
| `denied` | 19 | 20 | 95% |
| کل | 79 | 80 | 98.75% |

Macro F1 برابر 98.75% است.

تنها خطای routing:

- ID: `route-denied-002`
- سؤال: `List every policy stored in CRM.`
- expected: `denied`
- actual: `retrieval_only`
- HTTP: 200

علت این مورد، تشخیص‌ندادن عبارت bulk access با واژه مفرد `policy` در این phrasing است. این خطا مستقل از CRM و Guardrail است.

## 5. Operational End-to-End Success

در این گزارش Strict End-to-End Success یعنی دو شرط هم‌زمان برقرار باشند:

1. درخواست به route مورد انتظار برود.
2. اجرای downstream با HTTP قابل‌قبول و بدون خطای سیستمی تمام شود.

| مسیر | Strict E2E | Downstream completion |
| --- | ---: | ---: |
| `crm_only` | 20/20، 100% | 20/20، 100% |
| `retrieval_only` | 20/20، 100% | 20/20، 100% |
| `combined` | 20/20، 100% | 20/20، 100% |
| `denied` | 19/20، 95% | 20/20، 100% |
| کل | 79/80، 98.75% | 80/80، 100% |

تفاوت این دو ستون همان یک سؤال misroute شده است: سرویس پاسخ سالم HTTP 200 داده، اما چون درخواست باید denied می‌شد، Strict E2E آن را شکست محسوب می‌کند.

## 6. وضعیت HTTP و خطاها

| HTTP status | تعداد قبلی | تعداد جدید |
| --- | ---: | ---: |
| 200 | 10 | 71 |
| 403 | 9 | 9 |
| 503 | 61 | 0 |

| خطا | تعداد قبلی | تعداد جدید |
| --- | ---: | ---: |
| `CRM_REQUEST_FAILED` | 40 | 0 |
| `GUARDRAIL_INVALID_OUTPUT` | 21 | 0 |

Operational End-to-End قبلی 19/80 یا 23.75% بود. مقدار جدید Strict E2E برابر 79/80 یا 98.75% است؛ یعنی 60 مورد بیشتر و 75 واحد درصد بهبود.

## 7. Latency اجرای جدید

- Mean: حدود 2,412.9 ms
- P50: حدود 283.6 ms
- P95: حدود 6,546.5 ms
- Max: حدود 9,340.2 ms

## 8. تست‌های کد

مجموعه تست‌های مرتبط با CRM routing، Guardrail readiness، API boot و dataset اجرا شد:

`33 passed`

هشدارهای باقی‌مانده فقط deprecation warning کتابخانه‌های NeMo/LangChain هستند و شکست تست محسوب نمی‌شوند.

## نتیجه نهایی

مشکل عملیاتی CRM و Output Guardrail رفع شده است. هر چهار مسیر از نظر downstream عملیاتی 100% پاسخ داده‌اند. تنها مورد باقی‌مانده یک خطای مستقل routing در bulk-policy denial است؛ بنابراین Routing Accuracy و Strict End-to-End هر دو 98.75% هستند، نه 100%.
