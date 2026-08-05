# E2E MCP Pipeline Report

Started: 2026-07-07T01:01:19.3477298+02:00


## Phase 0: Git- und Ausgangszustand


## git status --short

```text
COMMAND: git status --short
 M requirements.txt
 M src/api/rag_service.py
?? reports/.latest_e2e_mcp_paths.txt
?? reports/e2e_mcp_pipeline_log_20260707_010119.txt
?? reports/e2e_mcp_pipeline_report_20260707_010119.md
?? scripts/test_mcp_retrieval_client.py
?? src/mcp_servers/

EXIT_CODE: 0
```


## Python Version

```text
COMMAND: .venv\Scripts\python.exe --version
Python 3.12.2

EXIT_CODE: 0
```


## Dependency Versions

```text
COMMAND: .venv\Scripts\python.exe -c 
import importlib.metadata as m
for p in ['fastmcp','langchain-mcp-adapters','chromadb','langchain','pandas','openpyxl','python-docx','pdfplumber']:
    try:
        print(f'{p}=={m.version(p)}')
    except Exception as exc:
        print(f'{p} ERROR {type(exc).__name__}: {exc}')

fastmcp==3.4.0
langchain-mcp-adapters==0.3.0
chromadb==1.3.5
langchain==1.1.2
pandas==2.3.3
openpyxl==3.1.5
python-docx==1.2.0
pdfplumber==0.11.10

EXIT_CODE: 0
```


## Phase 1: Preflight Checks


## fastmcp import

```text
COMMAND: .venv\Scripts\python.exe -c import fastmcp; print('FASTMCP_OK', getattr(fastmcp, '__version__', 'unknown'))
FASTMCP_OK 3.4.0

EXIT_CODE: 0
```


## langchain-mcp-adapters import

```text
COMMAND: .venv\Scripts\python.exe -c import langchain_mcp_adapters; print('MCP_ADAPTERS_OK')
MCP_ADAPTERS_OK

EXIT_CODE: 0
```


## retrieve_documents_for_tool import

```text
COMMAND: .venv\Scripts\python.exe -c from src.api.rag_service import retrieve_documents_for_tool; print('RETRIEVE_IMPORT_OK', callable(retrieve_documents_for_tool))
RETRIEVE_IMPORT_OK True

EXIT_CODE: 0
```


## retrieval_server import

```text
COMMAND: .venv\Scripts\python.exe -c import src.mcp_servers.retrieval_server as s; print('MCP_SERVER_IMPORT_OK', hasattr(s, 'mcp'), hasattr(s, 'retrieve_documents'))
MCP_SERVER_IMPORT_OK True True

EXIT_CODE: 0
```


## test_mcp_retrieval_client import

```text
COMMAND: .venv\Scripts\python.exe -c import scripts.test_mcp_retrieval_client as s; print('MCP_CLIENT_SCRIPT_IMPORT_OK')
MCP_CLIENT_SCRIPT_IMPORT_OK

EXIT_CODE: 0
```


## py_compile rag_service

```text
COMMAND: .venv\Scripts\python.exe -m py_compile src\api\rag_service.py

EXIT_CODE: 0
```


## py_compile retrieval_server

```text
COMMAND: .venv\Scripts\python.exe -m py_compile src\mcp_servers\retrieval_server.py

EXIT_CODE: 0
```


## py_compile test_mcp_retrieval_client

```text
COMMAND: .venv\Scripts\python.exe -m py_compile scripts\test_mcp_retrieval_client.py

EXIT_CODE: 0
```


Preflight status: **True**


## Phase 2: Chroma-/Index-Diagnose


## Chroma and source diagnosis

```text
COMMAND: .venv\Scripts\python.exe -c 
from pathlib import Path
from collections import Counter
from datetime import datetime
import json
import chromadb
persist = Path('data/processed/vectorstores/chroma_db')
print('PERSIST_EXISTS', persist.exists())
print('PERSIST_PATH', persist.resolve())
if persist.exists():
    for p in sorted(persist.iterdir(), key=lambda x: x.name.lower()):
        print('PERSIST_ITEM', 'DIR' if p.is_dir() else 'FILE', p.name, p.stat().st_size if p.is_file() else '')
client = chromadb.PersistentClient(path=str(persist))
for name in ['insurance_rag_collection','insuranceqa_collection']:
    try:
        col = client.get_collection(name)
        print('COLLECTION_COUNT', name, col.count())
    except Exception as exc:
        print('COLLECTION_ERROR', name, type(exc).__name__, str(exc))
from src.api.rag_service import PDF_DIRECTORY, get_source_files
files = get_source_files(PDF_DIRECTORY)
print('PDF_DIRECTORY', PDF_DIRECTORY)
print('SOURCE_FILE_COUNT', len(files))
print('SOURCE_EXTENSIONS', dict(sorted(Counter(Path(f).suffix.lower() for f in files).items())))
hash_path = Path('data/processed/caches/pdf_hashes.json')
print('PDF_HASH_EXISTS', hash_path.exists())
if hash_path.exists():
    data = json.loads(hash_path.read_text(encoding='utf-8'))
    print('PDF_HASH_COUNT', len(data) if isinstance(data, dict) else 'not-dict')
    print('PDF_HASH_LAST_WRITE', datetime.fromtimestamp(hash_path.stat().st_mtime).isoformat(sep=' ', timespec='seconds'))

PERSIST_EXISTS True
PERSIST_PATH C:\Users\mirae\MVA_Versicherung_Langchain_main\data\processed\vectorstores\chroma_db
PERSIST_ITEM DIR 2b9250ae-3281-4818-9454-ae4d1af2b1c6 
PERSIST_ITEM DIR 3a980506-c863-439d-82c9-1bc7d70c953e 
PERSIST_ITEM DIR 46c84356-a45b-4e99-8f45-b815b3e9d0ac 
PERSIST_ITEM DIR 7eb231e8-af45-4638-a9da-9c2664a9ce6d 
PERSIST_ITEM DIR 89000920-4c98-44af-991e-e5f19f75fc96 
PERSIST_ITEM DIR b6c9cc8d-7e9c-4184-b839-6772dc45ab65 
PERSIST_ITEM FILE chroma.sqlite3 98697216
PERSIST_ITEM DIR d4e9c0ac-31f1-4105-a589-b4ab6883547c 
COLLECTION_COUNT insurance_rag_collection 0
COLLECTION_COUNT insuranceqa_collection 1248
PDF_DIRECTORY data\raw\pdfs
SOURCE_FILE_COUNT 28
SOURCE_EXTENSIONS {'.pdf': 28}
PDF_HASH_EXISTS True
PDF_HASH_COUNT 28
PDF_HASH_LAST_WRITE 2026-05-26 04:23:43

EXIT_CODE: 0
```


## Phase 3: Sicherheitspr?fung vor Reindex


## Relevant build_vectorstore code

```python
1128: def build_vectorstore(
1129:     all_splits: List[Document],
1130:     embeddings,
1131:     force_reindex: bool = False,
1132: ) -> ChromaVectorStore:
1133:     chromadb_module = _get_chromadb_module()
1134:     chroma_cls = _get_chroma_cls()
1135:     client = chromadb_module.PersistentClient(path=str(SETTINGS.storage.chroma_persist_directory))
1136:     collection = client.get_or_create_collection(name=SETTINGS.storage.collection_name)
1137: 
1138:     vector_store = chroma_cls(
1139:         client=client,
1140:         collection_name=SETTINGS.storage.collection_name,
1141:         embedding_function=embeddings,
1142:         persist_directory=str(SETTINGS.storage.chroma_persist_directory),
1143:     )
1144: 
1145:     needs_reindex = force_reindex or collection.count() == 0
1146: 
1147:     if not needs_reindex and collection.count() > 0:
1148:         try:
1149:             vector_store.similarity_search("dimension check", k=1)
1150:         except Exception as exc:
1151:             msg = str(exc).lower()
1152:             if "expecting embedding with dimension" in msg or "dimension" in msg:
1153:                 print(f"Detected embedding dimension mismatch. Reindex required: {exc}")
1154:                 needs_reindex = True
1155:             else:
1156:                 raise
1157: 
1158:     if needs_reindex:
1159:         if collection.count() > 0:
1160:             client.delete_collection(name=SETTINGS.storage.collection_name)
1161:             client.create_collection(name=SETTINGS.storage.collection_name)
1162:             vector_store = chroma_cls(
1163:                 client=client,
1164:                 collection_name=SETTINGS.storage.collection_name,
1165:                 embedding_function=embeddings,
1166:                 persist_directory=str(SETTINGS.storage.chroma_persist_directory),
1167:             )
1168: 
1169:         batch_size = 1000
1170:         print(f"Info: Reindexing Chroma with {len(all_splits)} chunks (batch_size={batch_size}).")
1171: 
1172:         for start in range(0, len(all_splits), batch_size):
1173:             batch = all_splits[start:start + batch_size]
1174:             vector_store.add_documents(documents=batch)
1175:         pdf_files = get_pdf_files(PDF_DIRECTORY)
1176:         if pdf_files:
1177:             save_pdf_hashes(get_pdf_hashes(pdf_files))
1178: 
1179:     return vector_store
```


## Relevant initialize code

```python
1598:     def initialize(self, force_reindex: bool = False) -> Dict[str, Any]:
1599:         with self._init_lock:
1600:             if self.components is not None and not force_reindex:
1601:                 return self.components
1602: 
1603:             answer_model_warning = build_answer_model_warning(self.settings)
1604:             if answer_model_warning:
1605:                 print(f"Warning: {answer_model_warning}")
1606: 
1607:             os.makedirs(PDF_DIRECTORY, exist_ok=True)
1608:             source_files = get_source_files(PDF_DIRECTORY)
1609:             use_insuranceqa = os.getenv("USE_INSURANCEQA_DATA", "").strip().lower() in {
1610:                 "1",
1611:                 "true",
1612:                 "yes",
1613:                 "y",
1614:                 "on",
1615:             }
1616:             insuranceqa_mode = os.getenv("INSURANCEQA_RETRIEVAL_MODE", "merge").strip().lower()
1617: 
1618:             # PDF layer is optional when we explicitly run in InsuranceQA-only modes.
1619:             vector_store: Optional[ChromaVectorStore] = None
1620:             hybrid_retriever: Callable[[str, Optional[int]], List[Document]]
1621: 
1622:             if source_files:
1623:                 all_splits = load_and_split_documents(source_files)
1624:                 embeddings = initialize_embeddings()
1625: 
1626:                 reindex_required = force_reindex or embedding_model_has_changed()
1627:                 vector_store = self.build_vectorstore(all_splits, embeddings, force_reindex=reindex_required)
1628:                 hybrid_retriever = self.build_retriever(vector_store, all_splits)
```


Sicherheitsbewertung: OK: build_vectorstore(force_reindex=True) deletes only SETTINGS.storage.collection_name via client.delete_collection(name=...), not the whole persist directory.


## Phase 4: Tempor?re Multi-Source-Testdateien


## Create temporary multi-source files

```text
COMMAND: .venv\Scripts\python.exe -c 
from pathlib import Path
import pandas as pd
from docx import Document
from src.api.rag_service import PDF_DIRECTORY
base = Path(PDF_DIRECTORY)
base.mkdir(parents=True, exist_ok=True)
(base / 'test_multisource_txt.txt').write_text('MVA_UNIQUE_TXT_74291\nThe special deductible for the TXT test policy is 777 euros.\n', encoding='utf-8')
(base / 'test_multisource_table.csv').write_text('policy_code,premium,deductible\nMVA_UNIQUE_CSV_74291,123,456\n', encoding='utf-8')
df = pd.DataFrame([{'plan_code':'MVA_UNIQUE_XLSX_74291','waiting_period':'30 days','coverage_limit':9000}])
with pd.ExcelWriter(base / 'test_multisource_excel.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Plans', index=False)
doc = Document()
doc.add_paragraph('MVA_UNIQUE_DOCX_74291')
doc.add_paragraph('The DOCX policy includes roadside assistance after 48 hours.')
doc.save(base / 'test_multisource_docx.docx')
for name in ['test_multisource_txt.txt','test_multisource_table.csv','test_multisource_excel.xlsx','test_multisource_docx.docx']:
    p = base / name
    print('CREATED', p, p.exists(), p.stat().st_size)

CREATED data\raw\pdfs\test_multisource_txt.txt True 84
CREATED data\raw\pdfs\test_multisource_table.csv True 62
CREATED data\raw\pdfs\test_multisource_excel.xlsx True 5010
CREATED data\raw\pdfs\test_multisource_docx.docx True 36645

EXIT_CODE: 0
ELAPSED_SECONDS: 10.9
```


## Phase 5: Kontrollierter Full Reindex


## Controlled initialize_pipeline force_reindex

```text
TIMEOUT after 7200.0s
C:\Users\mirae\MVA_Versicherung_Langchain_main\src\api\rag_service.py:1121: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface package and should be used instead. To use it run `pip install -U `langchain-huggingface` and import as `from `langchain_huggingface import HuggingFaceEmbeddings``.
  return embeddings_cls(

```


Phase 5 failed. Further test phases will not run; cleanup still required.


## Phase 9: Cleanup


```text
REMOVED C:\Users\mirae\MVA_Versicherung_Langchain_main\data\raw\pdfs\test_multisource_txt.txt
REMOVED C:\Users\mirae\MVA_Versicherung_Langchain_main\data\raw\pdfs\test_multisource_table.csv
REMOVED C:\Users\mirae\MVA_Versicherung_Langchain_main\data\raw\pdfs\test_multisource_excel.xlsx
REMOVED C:\Users\mirae\MVA_Versicherung_Langchain_main\data\raw\pdfs\test_multisource_docx.docx
COLLECTION_COUNTS_AFTER_CLEANUP {'insurance_rag_collection': 6000, 'insuranceqa_collection': 1248}
TEMP_SOURCES_STILL_IN_INDEX []
```


## Phase 10: Abschlussbericht

### Gesamtstatus

**FAIL**

Der Test wurde kontrolliert abgebrochen, weil Phase 5 `initialize_pipeline(force_reindex=True)` nach dem gesetzten Timeout nicht sauber abgeschlossen hat. Gem?? Auftrag wurden Phase 6 Direct Retrieval, Phase 7 MCP Tool Call und Phase 8 run_rag danach nicht mehr ausgef?hrt.

### Phasenstatus

- Preflight: PASS
- Chroma Diagnose: PASS
- Reindex: FAIL / TIMEOUT
- Direct Retrieval: SKIPPED
- MCP Tool Discovery: SKIPPED in diesem Lauf, weil Reindex nicht sauber abgeschlossen wurde
- MCP Tool Call: SKIPPED
- run_rag: SKIPPED
- Cleanup: PASS

### Collection Counts

Vor Reindex:

- insurance_rag_collection: 0
- insuranceqa_collection: 1248

Nach Timeout und Cleanup:

- insurance_rag_collection: 6000
- insuranceqa_collection: 1248

Interpretation: Die Haupt-Collection ist nicht mehr leer, aber der Reindex wurde nicht sauber abgeschlossen. Der Count von 6000 deutet sehr wahrscheinlich auf einen partiellen Indexstand hin. Die tempor?ren Multi-Source-Testdateien wurden nicht im Index gefunden.

### Multi-Source-Ergebnisse

- TXT: NOT TESTED, Reindex nicht sauber abgeschlossen
- CSV: NOT TESTED, Reindex nicht sauber abgeschlossen
- XLSX: NOT TESTED, Reindex nicht sauber abgeschlossen
- DOCX: NOT TESTED, Reindex nicht sauber abgeschlossen
- PDF Table: NOT TESTED, Reindex nicht sauber abgeschlossen

### MCP-Ergebnisse

- Tool sichtbar: Nicht erneut getestet in diesem Lauf
- Tool Call erfolgreich: Nicht getestet
- Antwort/Document erhalten: Nein, weil Tool-Call-Phase ?bersprungen wurde

### Offene Probleme

1. `initialize_pipeline(force_reindex=True)` lief l?nger als 2 Stunden und wurde vom Testcontroller als Timeout behandelt.
2. W?hrend des Reindex war nur eine Deprecation-Warnung sichtbar; durch Output-Buffering kamen keine Fortschrittszeilen rechtzeitig zur?ck.
3. `insurance_rag_collection` hat jetzt 6000 Eintr?ge, aber der Reindex ist wahrscheinlich partiell und deshalb nicht als sauberer Zielzustand zu bewerten.
4. Die tempor?ren Testdateien wurden nach Cleanup nicht in der Collection gefunden. Wahrscheinlich wurde der Reindex vor Erreichen der letzten Source-Dateien abgebrochen.

### Dateien ge?ndert/erstellt

Dieser Testlauf hat keine Projektlogik ge?ndert. Erstellt wurden:

- `reports\e2e_mcp_pipeline_report_20260707_010119.md`
- `reports\e2e_mcp_pipeline_log_20260707_010119.txt`
- `reports/.latest_e2e_mcp_paths.txt`
- `reports/.latest_e2e_state.json`

Bereits vor dem Test vorhandene uncommitted ?nderungen laut aktuellem Git-Status:

```text
 M requirements.txt
 M src/api/rag_service.py
?? reports/.latest_e2e_mcp_paths.txt
?? reports/.latest_e2e_state.json
?? reports/e2e_mcp_pipeline_log_20260707_010119.txt
?? reports/e2e_mcp_pipeline_report_20260707_010119.md
?? scripts/test_mcp_retrieval_client.py
?? src/mcp_servers/
```

### Tempor?re Dateien entfernt

```text
C:\Users\mirae\MVA_Versicherung_Langchain_main\data\raw\pdfs\test_multisource_txt.txt
C:\Users\mirae\MVA_Versicherung_Langchain_main\data\raw\pdfs\test_multisource_table.csv
C:\Users\mirae\MVA_Versicherung_Langchain_main\data\raw\pdfs\test_multisource_excel.xlsx
C:\Users\mirae\MVA_Versicherung_Langchain_main\data\raw\pdfs\test_multisource_docx.docx
```

### Empfehlung f?r den n?chsten Schritt

Vor einem neuen Full-E2E-Test sollte der Reindex separat stabil gemacht werden: entweder mit besserem Fortschrittslogging pro Datei/Batch, l?ngerer Laufzeit ohne Controller-Timeout oder zuerst mit einem kleinen isolierten Source-Set. Danach sollte die Collection vollst?ndig und ?berpr?fbar neu aufgebaut werden, bevor Direct Retrieval und MCP Tool Call erneut getestet werden.

Finished: 2026-07-07T03:07:03

