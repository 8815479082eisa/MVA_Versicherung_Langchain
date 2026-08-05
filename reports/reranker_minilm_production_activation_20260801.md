# MiniLM-Reranker: Produktionsaktivierung (2026-08-01)

## Ergebnis

Die produktive lokale Konfiguration verwendet jetzt
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Das Modell wurde im laufenden
Backend-Container auf CPU als `BertForSequenceClassification` geladen. FP16 ist
deaktiviert. Ein Rollback war nicht erforderlich.

## Konfiguration und Laufzeit

| Prüfung | Ergebnis |
|---|---|
| Konfigurierter Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Tatsächlich geladener Modellname | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Adapter | `FlagEmbedding` / `BaseReranker` |
| Kernmodell | `BertForSequenceClassification` |
| Gerät | CPU |
| FP16 | `false` |
| Modell-Maximallänge | 512 Tokens laut Modellkonfiguration |
| Backend-Zustand | healthy |
| CRM-Zustand | healthy |

Der produktive Factory-Pfad verwendet weiterhin den vorhandenen
`FlagEmbedding`-Adapter. `sentence-transformers` ist installiert, wird für
dieses Modell im produktiven Pfad aber nicht benötigt. Das Modell wird beim
ersten Retrieval lazy geladen und danach in der Backend-Instanz wiederverwendet.
Die Produktionsfactory setzt aktuell keine eigene Batchgröße; beim geprüften
E2E-Lauf wurden 11 Kandidaten gemeinsam gererankt.

## Relevante Tests

Ausgeführt:

```text
.\.venv\Scripts\python.exe -m pytest tests/unit/test_current_policy.py tests/unit/test_crm_orchestration.py tests/integration/test_crm_api_routing.py -q
```

Ergebnis: **22 passed, 0 failed**. Die 22 Warnungen betreffen vorhandene
Pydantic-Deprecations. Für diese Aktivierung wurden keine Python-Dateien
geändert und keine Abhängigkeiten aktualisiert.

## Reale End-to-End-Prüfung

Anfrage an `POST http://127.0.0.1:8000/api/ask`:

```text
What is Lara Neumann's current motor insurance policy, including the policy
number, coverage type, deductible and annual premium, and is windscreen damage
generally covered according to the available insurance documents?
```

Zwei reale Läufe mit CRM, bestehendem Chroma-Index, RAG, MiniLM und OpenAI
lieferten HTTP 200 und die Route `combined`.

CRM-Ergebnis nach dem deterministischen Current-Policy-Filter:

- Police: `TEST-KFZ-2026-1003`
- Deckungsart: `Partial Coverage`
- Selbstbeteiligung: `300 EUR`
- Jahresprämie: `720 EUR`
- Stichtag: `2026-08-01`
- Alte Police `TEST-KFZ-2026-1001` ausgewählt: nein

Die Dokumentantwort verwies auf Seite 14 der
`motor-vehicle-insurance-sti.pdf`. Dort wird Glasdeckung für unfreiwillige
Bruch- und Unfallschäden an Front-/Heckscheiben und Seitenfenstern beschrieben.
Die Antwort ist im Ergebnis belegt. Einschränkung: Die Antwort verknüpft die
Begründung sprachlich teilweise mit `GlassPlus`, obwohl die direkte
Windschutzscheibenregel bereits unter der Basis-Glasdeckung steht. Außerdem
enthielt der ausgewählte Chunk von Seite 14 nicht den Anfang des unmittelbar
einschlägigen Abschnitts K2.1.5; die zitierte PDF-Seite selbst enthält ihn.

## Laufzeiten

| Messung | Kaltlauf | Warmlauf |
|---|---:|---:|
| HTTP-Wandzeit | 24,124 s | 11,498 s |
| Gemeldete API-Gesamtzeit | 23,860 s | 11,265 s |
| Reranking | nicht separat erfasst | 3,793 s |
| Retrieval gesamt | nicht separat erfasst | 2,232 s |
| Antwortgenerierung | nicht separat erfasst | 3,099 s |

Der Warmlauf rerankte 11 Kandidaten und gab 5 Dokumente aus. Als rein
indikative historische Referenz benötigte BGE im ähnlichsten vorhandenen
Fixed-E2E-Bericht etwa 12,9–13,3 s nur für das Reranking. Das ist kein streng
kontrollierter A/B-Vergleich, weil Zeitpunkt, Hostzustand und Kandidatenmenge
nicht vollständig identisch waren. Der isolierte Post-Filter-Benchmark bleibt
die belastbarere Modellvergleichsbasis: MiniLM erzielte dort 96,9 % Top-1 und
100 % für die englischen Fälle bei 467,8 ms Median; der Lara-Pflichtfall wurde
nach dem Metadatenfilter bestanden.

## Rollback

Kein Rollback erforderlich. Falls später ein kritisches Problem auftritt:

1. In `.env` `RERANKER_MODEL=BAAI/bge-reranker-base` setzen.
2. `RERANKER_USE_FP16=false` beibehalten.
3. Backend neu starten:

```text
docker compose --env-file .env.crm -f docker/docker-compose.yml -f docker/docker-compose.crm.yml restart backend
```

4. Geladenen Modellnamen und einen realen Lara-Neumann-Lauf erneut prüfen.

## Geänderte Dateien

- `.env`: Reranker auf MiniLM umgestellt; FP16 bleibt deaktiviert.
- `reports/reranker_minilm_production_activation_20260801.md`: dieser Bericht.

Es wurden keine Collections verändert oder gelöscht, kein Reindex ausgeführt
und kein produktiver Python-Code geändert.
