# گزارش Full End-to-End نسخه نهایی سیستم ـ ۸۰ Case

## ۱. دامنه، روش و یکپارچگی داده

این ارزیابی یک اجرای کاملاً جدید روی هر ۸۰ Query فایل `data/benchmarks/routing/routing_eval_80.jsonl` است. تمام Queryها از API واقعی `/api/ask` وارد Pipeline شدند و پاسخ ذخیره‌شده‌ای از Runهای قبلی reuse نشد. اجرای API از ساعت 10:24:07 تا 10:32:24 UTC در 23 اوت 2026 انجام شد.

- Dataset SHA-256: `1a65cf29c6ca11304a3265c7ce21eb38624b8ee81c195a01c1e06c0a63cf91d3`
- Reference Specification SHA-256: `430004953643d7bb1bfd7306f2751abca93567d948b14d08224d9e70335e9677`
- Reference Specification Version: `1.1.0`
- روش: `automated_reference_based_technical_validation`
- Human Validation: انجام نشده (`human_validated=false`)

پیش از اجرا، Backend، Pipeline، CRM، Embedding/Retrieval، LLM، Answer Model و Guardrail از طریق `/health` آماده گزارش شدند. یک warm-up خارج از Dataset نیز اجرای واقعی CrossEncoder Reranker، Answer Generation، Groundedness، Output Safety و دسترسی read-only به EspoCRM را تأیید کرد. Frontend جزو مسیر مستقیم این آزمون Backend API نبود.

## ۲. Overall Metrics

| گروه | Metric | نتیجه Run جدید |
|---|---|---:|
| Routing | Accuracy | **80/80 = 100%** |
| Routing | Macro-F1 | **100%** |
| Operational | Strict E2E، یعنی route صحیح و downstream کامل | **80/80 = 100%** |
| Operational | Downstream Operational Completion | **80/80 = 100%** |
| Operational | Collection error / HTTP 5xx / timeout | **0 / 0 / 0** |
| Answer Quality | Automated Overall Pass | **73/80 = 91.25%**؛ Wilson 95% CI: 83.02%–95.70% |
| CRM | Mean case-level CRM Fact Recall | **97.50%** |
| CRM | Micro Fact Recall | **68/70 = 97.14%** |
| Retrieval | Expected file hit in Reranker/final generation context | **39/40 = 97.50%** |
| Retrieval | Final Expected Document Presence | **35/40 = 87.50%** |
| Retrieval | Mean Retrieval Expected-File Precision | **93.75%** |
| Answer | Mean Requirement Recall | **85.00%** |
| Answer | Micro Requirement Recall | **54/65 = 83.08%** |
| Citation | Expected-source citation rate | **35/40 = 87.50%** |
| Citation | Citation Expected-File Precision | **97.22%** |
| Citation | Returned-Source-Link Precision | **100%** |
| Citation | Claim-Coverage Proxy | **80.55%** |
| Claim | Mean Claim Support Rate | **65.32%** |
| Groundedness | Pass Rate در Caseهای دارای score | **56/60 = 93.33%** |
| Groundedness | Mean / Median Score | **90.46% / 91.80%** |
| Safety | Denied routing accuracy | **20/20 = 100%** |
| Safety | Safe Response Rate | **20/20 = 100%** |
| Safety | No-Data-Access Rate | **20/20 = 100%** |
| Performance | Mean / p50 / p95 / max | **6.056 / 3.860 / 15.281 / 20.368 s** |

نتیجه 73/80 نشان‌دهنده عبور از gate خودکار مرجع است؛ این عدد معادل صحت انسانی 91.25 درصد نیست.

## ۳. Per-Route Metrics

| Expected route | Cases | Strict E2E | Automated Pass | CRM Recall | Requirement Recall | Expected Document Presence | Claim Support | Groundedness Pass | Mean latency | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CRM-only | 20 | 100% | **20/20 = 100%** | 100% | n/a | n/a | n/a | 100% | 0.553 s | 0.545 s | 0.814 s | 0.892 s |
| retrieval_only | 20 | 100% | **16/20 = 80%** | n/a | 85% | 80% | 68.99% | 85% | 12.398 s | 11.994 s | 16.307 s | 20.368 s |
| Combined | 20 | 100% | **17/20 = 85%** | 95% | 85% | 95% | 61.64% | 95% | 11.189 s | 10.743 s | 15.327 s | 18.000 s |
| Denied | 20 | 100% | **20/20 = 100%** | n/a | n/a | n/a | n/a | n/a | 0.085 s | 0.077 s | 0.131 s | 0.217 s |

## ۴. Routing

### Precision، Recall و F1

| Route | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| CRM-only | 20 | 0 | 0 | 100% | 100% | 100% |
| retrieval_only | 20 | 0 | 0 | 100% | 100% | 100% |
| Combined | 20 | 0 | 0 | 100% | 100% | 100% |
| Denied | 20 | 0 | 0 | 100% | 100% | 100% |

Macro-F1 برابر 100 درصد و Route mismatch برابر صفر بود.

### Confusion Matrix

ردیف‌ها Expected و ستون‌ها Actual هستند.

| Expected \ Actual | CRM-only | retrieval_only | Combined | Denied |
|---|---:|---:|---:|---:|
| CRM-only | **20** | 0 | 0 | 0 |
| retrieval_only | 0 | **20** | 0 | 0 |
| Combined | 0 | 0 | **20** | 0 |
| Denied | 0 | 0 | 0 | **20** |

## ۵. Operational End-to-End

| Route | HTTP 200 | HTTP 206 | HTTP 403 | System error | Operational Completion |
|---|---:|---:|---:|---:|---:|
| CRM-only | 20 | 0 | 0 | 0 | 100% |
| retrieval_only | 19 | 1 | 0 | 0 | 100% |
| Combined | 19 | 1 | 0 | 0 | 100% |
| Denied | 10 | 0 | 10 | 0 | 100% |
| مجموع | **68** | **2** | **10** | **0** | **100%** |

دو پاسخ `route-rag-002` و `route-combined-011` با status 206/partial پایان یافتند. `route-combined-011` با وجود status فنی partial، تمام معیارهای reference-based را پاس کرد؛ `route-rag-002` از نظر Answer Quality ناموفق بود. ده پاسخ 403، Denied صریح و مورد انتظار بودند و ده Denied دیگر به‌صورت پاسخ کنترل‌شده HTTP 200 برگشتند.

Strict E2E در این گزارش مطابق Run قبلی برابر است با «Route صحیح و پایان عملیاتی معتبر». بنابراین پاسخ Partial کنترل‌شده می‌تواند Operational Success باشد، ولی لزوماً Answer-Quality Pass نیست.

## ۶. CRM Quality

از ۴۰ Case دارای CRM reference، در ۳۹ Case همه Factهای مورد انتظار در پاسخ وجود داشت. از ۷۰ Fact مرجع، ۶۸ Fact بازیابی و منعکس شدند.

| Reference table | Expected facts | Present | Recall |
|---|---:|---:|---:|
| Claims | 17 | 16 | 94.12% |
| Contacts | 1 | 1 | 100% |
| Policies | 52 | 51 | 98.08% |
| مجموع | 70 | 68 | 97.14% |

تنها شکست CRM در `route-combined-002` رخ داد. Ground-truth همه ۴۰ Case از CSV قابل resolve بود، اما Dataset برچسب مستقلی برای «customer/policy/claim selection correctness» به‌صورت یک Metric مستقل ندارد. بنابراین این مفهوم جداگانه جعل نشده و فقط با Fact Recall و missing-factها گزارش شده است.

## ۷. Retrieval، RAG و Citation

در ۳۹ مورد از ۴۰ مسیر retrieval_only/Combined، فایل مورد انتظار هم در خروجی Reranker و هم در Generation Context وجود داشت. تنها miss واقعی Retrieval مربوط به `route-rag-010` بود که به‌جای brochure مورد انتظار، سند STI را وارد Context کرد.

Final Expected Document Presence برابر 35/40 بود. تفاوت 39/40 Context Hit با 35/40 Final Source Presence ناشی از چهار پاسخ fallback است: سند صحیح وارد Context شده بود، اما پاسخ نهایی «اطلاعات کافی نیست» تولید شد و Source/Citation نهایی حذف شد. بنابراین Retrieval Success و Final Source Handling دو Metric متفاوت‌اند.

Returned-Source-Link Precision برابر 100 درصد بود؛ یعنی Citationهای بازگشتی به Sourceهای بازگشتی لینک داشتند. اما Expected-source citation rate فقط 87.5 درصد بود و نشان می‌دهد link consistency به‌تنهایی کفایت citation را اثبات نمی‌کند.

## ۸. Answer Quality، Groundedness و Error Catalog

| Failure type | Count |
|---|---:|
| `requirement_recall_below_threshold` | 6 |
| `expected_document_source_missing` | 5 |
| `expected_document_citation_missing` | 5 |
| `groundedness_failed` | 4 |
| `crm_fact_incomplete_or_incorrect` | 1 |
| Route mismatch / operational failure / unsafe denial | 0 |

Groundedness برای ۶۰ Case غیر-Denied قابل محاسبه بود: میانگین 0.904648، میانه 0.918017، کمینه 0.631157 و بیشینه 1.0. چهار Failure عبارت بودند از `route-rag-002`، `route-rag-003`، `route-rag-005` و `route-combined-002`.

Precision، Recall و False-Acceptance Rate مربوط به Groundedness در این Run قابل محاسبه نیستند، زیرا Dataset ۸۰ موردی برای هر claim برچسب مستقل انسانی supported/unsupported ندارد. score و pass/fail سیستم قابل ثبت است، اما بدون Ground Truth دودویی مستقل، ساخت Confusion Matrix برای Groundedness علمی نیست.

## ۹. فهرست کامل Caseهای ناموفق

| ID | Expected | Actual | علت |
|---|---|---|---|
| `route-rag-002` | retrieval_only؛ توضیح پوشش سرقت و شرط‌بودن آن؛ سند motor STI | retrieval_only، HTTP 206؛ پاسخ «پاسخ کافیِ مستند تولید نشد» | سند صحیح وارد Context شد، اما Generation/Groundedness Repair نتوانست requirements سرقت را به پاسخ مستند تبدیل کند؛ Citation و Source نهایی حذف شدند. |
| `route-rag-003` | retrieval_only؛ تعریف collision و ارتباط آن با fully comprehensive؛ سند motor STI | retrieval_only، HTTP 200؛ fallback عدم کفایت شواهد | Retrieval hit بود، ولی پاسخ نهایی هر دو requirement را از دست داد و Groundedness=0.631157 شد. |
| `route-rag-005` | retrieval_only؛ liability، comprehensive و assistance در product sheet | retrieval_only، HTTP 200؛ fallback عدم کفایت شواهد | هر سه requirement پاسخ داده نشد؛ Groundedness=0.763461 و Source/Citation نهایی حذف شد. |
| `route-rag-010` | retrieval_only؛ خلاصه household contents و private liability از brochure مورد انتظار | retrieval_only، HTTP 200؛ پاسخ درباره private liability با citation سند STI | Requirement متنی پوشش داده شد و Groundedness پاس شد، اما Reranker فایل brochure مورد انتظار را وارد Context نکرد؛ در نتیجه Source/Citation expected-file ناموفق بود. |
| `route-combined-001` | Combined؛ پوشش glass/windscreen و parts به‌علاوه deductible=150 EUR | Combined، HTTP 200؛ CRM fact و نام دو source ارائه شد، اما توضیح پوشش وجود نداشت | CRM صحیح بود، ولی پاسخ deterministic فقط sourceها را فهرست کرد و هر دو requirement سندی را از دست داد. |
| `route-combined-002` | Combined؛ وضعیت claim، نوع policy و شرط پوشش glass | Combined، HTTP 200؛ fallback عدم کفایت شواهد | Requirement سندی، claim status و coverageType در پاسخ نهایی نبودند؛ Groundedness=0.680511. |
| `route-combined-020` | Combined؛ windscreen repair/replacement و deductible=500 EUR | Combined، HTTP 200؛ CRM fact و sourceها بدون حکم پوشش | deductible صحیح بود، اما requirement مربوط به repair/replacement برای safety reasons بیان نشد. |

## ۱۰. Performance و Stage Latency

Stage latency فقط برای stageهایی گزارش شده که instrumentation واقعی داشتند. تعداد `n` با مسیرهای واجد آن stage متفاوت است.

| Stage | n | Mean | p50 | p95 | Max |
|---|---:|---:|---:|---:|---:|
| Route planning | 60 | 0.684 ms | 0.599 ms | 1.224 ms | 1.788 ms |
| CRM | 40 | 225.765 ms | 233.517 ms | 328.733 ms | 362.438 ms |
| Guardrail | 40 | 859.497 ms | 792.053 ms | 1503.292 ms | 1530.584 ms |
| Retrieval | 40 | 393.240 ms | 84.525 ms | 3121.919 ms | 3478.448 ms |
| Reranking | 40 | 4790.368 ms | 4626.090 ms | 6479.034 ms | 7282.197 ms |
| Answer Generation | 40 | 3477.614 ms | 2921.909 ms | 7710.500 ms | 9258.517 ms |
| Groundedness | 40 | 636.323 ms | 578.179 ms | 1196.990 ms | 1279.134 ms |

Reranking بزرگ‌ترین سهم متوسط latency را دارد و پس از آن Answer Generation قرار می‌گیرد. Denied پیش از Pipeline سندی/CRM متوقف می‌شود؛ بنابراین stage timingهای RAG برای آن applicable نیستند.

## ۱۱. Safety و Denied

هر ۲۰ Denied Case به‌درستی تشخیص داده شد. Safe Response Rate و No-Data-Access Rate هر دو 100 درصد و `unsafe_or_missing_denial` برابر صفر بود. Diagnostics هیچ دسترسی CRM در Denied Caseها نشان نداد. تفاوت HTTP 403 و HTTP 200 به دو نوع پایان کنترل‌شده مربوط است و نه دسترسی غیرمجاز.

## ۱۲. Reliability و Regression

| Test suite | Passed | Failed | Skipped | Warnings |
|---|---:|---:|---:|---:|
| Unit | 281 + 13 subtests | 0 | 0 | 226 |
| Integration | 19 | 0 | 1 | 22 |

Warningهای اصلی مربوط به deprecation در NeMo Guardrails/Pydantic، sunset شدن `langchain-community` و deprecation در Starlette TestClient هستند؛ Failure تستی مشاهده نشد.

| Metric | Run تاریخی | Full Run جدید | تغییر |
|---|---:|---:|---:|
| Routing Accuracy | 98.75% | **100%** | +1.25 pp |
| Strict Operational E2E | 98.75% | **100%** | +1.25 pp |
| Downstream Operational Completion | 100% | **100%** | بدون تغییر |
| Automated Answer-Quality Pass | 45/80 = 56.25% | **73/80 = 91.25%** | +28 Case، +35 pp |

از ۳۵ Failure ارزیابی قدیمی، ۳۴ مورد اکنون پاس شدند و `route-rag-005` دوباره Fail شد. شش Case که در ارزیابی 45/80 قدیمی پاس بودند اکنون Fail شدند: `route-rag-002`، `route-rag-003`، `route-rag-010`، `route-combined-001`، `route-combined-002` و `route-combined-020`. بنابراین بهبود خالص ۲۸ Case است، اما Regression/Nondeterminism در Answer Generation و Groundedness Repair همچنان مشاهده می‌شود. نتیجه تاریخی 35/35 یک subset run مستقل بود و با Full Run جدید مخلوط نشده است؛ روی همان subset در Run جدید نتیجه 34/35 است.

## ۱۳. Not computable / Not applicable

- Groundedness Precision/Recall/FAR: فاقد claim-level Ground Truth مستقل و انسانی.
- صحت انسانی، کامل‌بودن تخصصی، usability، trust و خطر سوءبرداشت: بدون Human Validation/User Study قابل محاسبه نیست.
- Production Readiness، Load/Concurrency، هزینه و long-term stability: این Run ترتیبی ۸۰ Case یک Load Test یا Production Certification نیست.
- Customer/Policy/Claim selection accuracy به‌عنوان Metric مستقل: Dataset برچسب انتخاب مستقل ندارد؛ فقط Fact Recall و missing facts گزارش شد.
- Stage latency برای stageهایی که در یک route اجرا یا instrument نشده‌اند: n/a و نه صفر.

## ۱۴. متن علمی کوتاه برای فصل ۵ Masterarbeit

### متن فارسی

در ارزیابی جدید End-to-End، تمام ۸۰ پرسش مجموعه‌داده ثابت از طریق API واقعی سیستم اجرا شدند. دقت Routing، Macro-F1، Strict End-to-End Success و Downstream Operational Completion همگی به 100 درصد رسیدند و هیچ خطای سیستمی، timeout یا پاسخ HTTP 5xx مشاهده نشد. بااین‌حال، ارزیابی کیفیت پاسخ مبتنی بر Reference نتیجه 73 از 80 یا 91.25 درصد را نشان داد. هفت شکست باقی‌مانده عمدتاً به تبدیل‌نشدن شواهد بازیابی‌شده به پاسخ کامل، حذف Source/Citation در پاسخ fallback و یک Retrieval miss مربوط بودند. بنابراین موفقیت فنی و عملیاتی کامل، به معنای کیفیت کامل پاسخ نیست. این نتایج یک ارزیابی فنی خودکار هستند و بدون ارزیابی انسانی متخصص یا User Study نباید به‌عنوان Human Validation یا اثبات Production Readiness تفسیر شوند.

### Direkt verwendbarer deutscher Kurztext

In der erneuten End-to-End-Evaluation wurden alle 80 Anfragen des unveränderten Datensatzes über die reale API und die jeweils vorgesehenen Verarbeitungskomponenten ausgeführt. Routing Accuracy, Macro-F1, Strict End-to-End Success und Downstream Operational Completion erreichten jeweils 100 %, wobei weder Systemfehler noch Timeouts oder HTTP-5xx-Antworten auftraten. Die davon getrennte, automatisierte referenzbasierte Antwortbewertung ergab 73 von 80 bestandenen Fällen beziehungsweise 91,25 %. Die sieben verbleibenden Fehler waren überwiegend darauf zurückzuführen, dass vorhandene Retrieval-Evidenz nicht in eine vollständige, hinreichend gestützte Antwort überführt wurde, Quellenangaben nach einem Fallback fehlten oder das erwartete Dokument nicht in den finalen Kontext gelangte. Die Ergebnisse zeigen damit eine vollständige technische und operative Ausführbarkeit, jedoch keine vollständige fachliche Antwortqualität. Da weder eine unabhängige Expertenannotation noch eine Nutzerstudie Bestandteil dieses Laufs war, dürfen die Resultate nicht als Human Validation oder Nachweis der Produktionsreife interpretiert werden.

## ۱۵. نتیجه‌گیری

نسخه فعلی از نظر Routing و اجرای عملیاتی نسبت به Full E2E قبلی Regression ندارد و خطای Routing قبلی نیز رفع شده است. Answer Quality از 56.25 درصد به 91.25 درصد افزایش یافته، اما به دلیل هفت Failure و مشاهده نوسان بین Runها هنوز 100 درصد نیست. نتیجه صحیح این Run عبارت است از: **Technical/Operational Success = 100%، Automated Reference-Based Answer Quality = 91.25%، Human Validation = انجام نشده**.
