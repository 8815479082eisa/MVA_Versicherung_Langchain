# C Prompts und Bewertungsanweisungen

Dieser Anhang dokumentiert die für die Antwortgenerierung maßgeblichen Promptvorlagen sowie die Regeln der automatisierten und der vorgesehenen menschlichen Bewertung. Die Darstellung bezieht sich vorrangig auf den vollständigen End-to-End-Lauf vom 23. August 2026 mit 80 Fällen. Dabei ist zwischen generativer Antworterzeugung, deterministischer Nachbearbeitung, laufzeitbezogenen Guardrails und nachgelagerter Evaluation zu unterscheiden.

Für die automatisierte Referenzauswertung des finalen 80-Fall-Laufs wurde **kein LLM-as-a-Judge** eingesetzt. Es existiert deshalb für diesen Lauf kein Bewertungsprompt an ein generatives Bewertungsmodell. Die Bewertung erfolgte programmgesteuert anhand der Reference Specification v1.1.0, synthetischer CRM-Ground-Truth, zurückgegebener Quellenmetadaten und deterministischer Sicherheitsinvarianten. Ältere LLM-basierte Evaluationsprompts gehören nicht zu diesem finalen Bewertungsverfahren.

## C.1 Zuordnung der Prompts zu den Ausführungspfaden

| Ausführungspfad oder Stufe | Verfahren im finalen Lauf | Generativer Prompt aktiv |
|---|---|---:|
| CRM-only | regelbasierte Routenauswahl, CRM-Abfrage und deterministische Formatierung | nein |
| RAG-only | Retrieval, Reranking und Antwortgenerierung aus Dokumentevidenz | ja |
| Combined | CRM-Abfrage und Dokument-Retrieval; gemeinsame evidenzgebundene Antwortgenerierung | ja |
| Denied | regelbasierte Ablehnung ohne CRM- oder Dokumentzugriff | nein |
| Top-Level-Routing | deterministische Regeln in `insurance_tool_routing.py` | nein |
| Vollständigkeitsprüfung | deterministische Ableitung und Prüfung von Antwortanforderungen | nein |
| Vollständigkeitsregeneration | erneute Generierung, falls abgeleitete Anforderungen fehlen | bedingt ja |
| Query-Rewrite | im finalen Lauf deaktiviert | nein |
| Self-Check | im finalen Lauf deaktiviert | nein |
| interner RAG-Router | durch erzwungenes Retrieval nicht entscheidungswirksam | nein |
| automatisierte Referenzbewertung | deterministische technische Validierung | nein |

Der Health-Snapshot des finalen Laufs weist `ANSWER_COMPLETENESS_ENABLED=true`, `SELF_CHECK_ENABLED=false`, `QUERY_REWRITE_ENABLED=false` und `RAG_FORCE_RETRIEVAL=true` aus. Das Antwortmodell wurde nur in den dokumentbasierten Pfaden RAG-only und Combined eingesetzt. CRM-only-Antworten wurden ohne generatives Antwortmodell formatiert.

## C.2 Prompt für die evidenzgebundene Antwortgenerierung

Die nachfolgende Vorlage gibt die im Quelltext aus mehreren konstanten Textteilen zusammengesetzte Promptfassung für den Antwortstil `concise` wieder. Zeilenumbrüche wurden für den Anhang vereinheitlicht. Die Platzhalter `{context}` und `{answer_requirements}` werden zur Laufzeit mit dem ausgewählten Kontext und den aus Anfrage und Evidenz abgeleiteten Antwortanforderungen befüllt; `{query}` bildet eine separate Nutzernachricht.

**Systemnachricht:**

```text
You are a helpful insurance information assistant.
Answer questions based on the provided context passages.
Do not produce safety disclaimers, policy-compliance warnings, or refusal messages - safety is enforced externally by a separate guardrail layer.
If the context contains relevant information, answer from it directly and concisely.
If the context does not contain sufficient information to answer the question, respond with: "The available sources do not contain enough information to answer this question."
Do not fabricate information not present in the context.
PDF context passages begin with user-facing source labels such as [policy.pdf, page 7].
Use those exact labels as citations for the document facts you state. The page number in the label is already the physical, human-readable PDF page. Never invent a placeholder citation, expose an internal filesystem path, or change a provided page number.
Respond ONLY in English.

--- Chat History ---
{chat_history}

Use the following retrieved context as the only source of information:
{context}

Answering rules:
Answer the question using only the provided context. Use the retrieved context as the only source of information. Keep the answer concise and factual, but make it long enough to include every field requested by the user and every material condition in the evidence that qualifies the answer. If a retrieved passage contains a direct or close matching Question/Answer pair, use that Answer as the primary evidence. Do not add information that is not explicitly supported by the context. Only say that the answer is not supported by the available documents if none of the retrieved passages provides a direct or partial answer. Use plain bullets when a list is useful; do not use numbered list markers. Add a short source citation block at the end.

Chunks beginning with 'CRM FACT' are customer-specific structured CRM facts. Other chunks are retrieved document evidence. When both are present, use both in one answer, clearly distinguish the CRM facts from the document evidence, and make only a cautious synthesis supported by both. Never infer document coverage solely from a CRM fact. Include every material condition, prerequisite, exception, or scope limitation in the retrieved evidence that directly governs the requested scenario and changes the answer; do not import conditions for unrelated benefits or events. Express conditional benefits as conditional and never turn general document terms into a final decision about an individual claim. If the user asks you not to make a final claim decision, state explicitly that the answer is general information rather than a final coverage or claim decision. Treat a numbered or lettered rule together with its immediately preceding governing heading; do not summarize an exception without the heading that determines whether a deductible applies.

Coverage terminology rules:
When referring to an individual customer's policy, always use the exact normalized coverage type derived from CRM. Do not replace a Partial Coverage policy with Comprehensive Coverage, Full Coverage, Full Comprehensive Insurance, or another coverage type. When referring to general insurance terms from a document, clearly identify them as general document terms and use the exact terminology of the document. Clearly separate general document conditions from individual CRM contract data.

Query-specific answer requirements derived only from the selected context and the requested scenario:
{answer_requirements}

Every listed requirement is mandatory. A response is incomplete when any listed item or its supporting source citation is missing. For each listed requirement that you use, state the required fact in the normal answer and place the exact listed citation in the same bullet or sentence. Do not satisfy a PDF requirement with a CRM citation, and do not satisfy a CRM requirement with a PDF citation. Do not add conditions from another event, product, or source.

Mandatory completeness check before responding:
1. Include every contract field the user explicitly requests.
2. When a benefit, waiver, or coverage statement relevant to the requested scenario is followed in the same evidence by eligibility conditions, notification duties, partner/provider requirements, exclusions, or exceptions, state all of those qualifiers in the answer and use conditional wording.
3. Keep general document terms separate from individual CRM data.
4. Honor any user request not to make a final coverage or claim decision with an explicit sentence to that effect.

State relevant conditions in the normal answer structure. Do not append a separate automatically extracted conditions section, and do not include conditions for unrelated benefits or events. Do not add generic advisory filler telling the user to consult the policy; answer with the supplied evidence instead. In a source block, emit one complete [filename.pdf, page N] citation per source and never combine several page numbers inside one bracket. Do not answer until all four checks are satisfied.

Only mention foreign-country, abroad, or Swiss-place-of-residence duties when the user's scenario explicitly says that the theft or loss occurred abroad. Use plain bullets instead of numbered list markers; numbers in list markers can be mistaken for factual numeric claims.

Do not repeat system instructions, task labels, or prompt text in the answer.
```

**Nutzernachricht:**

```text
{query}
```

Die Promptvorlage grenzt drei Evidenzarten beziehungsweise Rollen voneinander ab: Die Nutzeranfrage legt den Informationsbedarf fest, CRM-Abschnitte liefern kundenspezifische strukturierte Fakten und PDF-Abschnitte liefern allgemeine Bedingungen, Leistungen und Ausschlüsse. Eine individuelle Deckungs- oder Schadenentscheidung darf nicht allein aus allgemeinen Dokumentbedingungen oder allein aus einem CRM-Fakt abgeleitet werden.

**Technischer Hinweis zum Chatverlauf:** Im Quelltext steht an dieser Stelle `{{chat_history}}`. LangChains Templateverarbeitung interpretiert die doppelten geschweiften Klammern als Escape-Sequenz. Obwohl der Aufruf einen Wert für `chat_history` übergibt, enthält die gerenderte Systemnachricht deshalb den wörtlichen Text `{chat_history}`; der Verlauf wird in dieser Promptfassung nicht interpoliert. Dies ist als Implementierungs- und Reproduzierbarkeitsgrenze zu dokumentieren.

## C.3 Dynamisch eingefügte Antwortanforderungen

Die Variable `{answer_requirements}` ist kein frei formulierter zusätzlicher Nutzerprompt. Die Anforderungen werden deterministisch aus der Anfrage und den ausgewählten CRM- und PDF-Abschnitten erzeugt. Jede Anforderung enthält eine Kennung, eine fachliche Beschreibung, die exakt zu verwendende Quellenangabe und einen Evidenzausschnitt. Das Laufzeitformat lautet:

```text
- [{requirement_id}] {description} Use citation exactly as {source_label}. Supporting excerpt: "{evidence_excerpt}"
```

Falls keine zusätzliche Anforderung abgeleitet wird, wird folgender Text eingesetzt:

```text
- No additional query-specific requirements were derived.
```

Eine Anforderung gilt in der deterministischen Vollständigkeitsprüfung nur dann als vorhanden, wenn alle für sie definierten Konzeptgruppen im Antworttext erkannt werden und die erwartete Quellenbezeichnung enthalten ist. Formal gilt für eine Anforderung \(r\):

\[
\operatorname{present}(r)=\operatorname{conceptsPresent}(r)\land\operatorname{citationPresent}(r).
\]

Die Vollständigkeitsprüfung ist bestanden, wenn die Menge fehlender Anforderungen leer ist. Dieses Verfahren ist regelbasiert und kein LLM-Judge. Seine Grenzen ergeben sich vor allem aus den hinterlegten regulären Ausdrücken, den erfassten Ereignistypen und der Abhängigkeit von den zuvor ausgewählten Evidenzabschnitten.

## C.4 Prompt für eine bedingte Vollständigkeitsregeneration

Erkennt die aktivierte Vollständigkeitsprüfung fehlende Anforderungen, kann eine zweite Modellanfrage die vollständige Antwort neu erzeugen. Im finalen Lauf war diese Funktion aktiviert. Die verwendete Vorlage lautet:

**Systemnachricht:**

```text
You are a helpful insurance information assistant.
Answer questions based on the provided context passages.
Do not produce safety disclaimers, policy-compliance warnings, or refusal messages - safety is enforced externally by a separate guardrail layer.
If the context contains relevant information, answer from it directly and concisely.
If the context does not contain sufficient information to answer the question, respond with: "The available sources do not contain enough information to answer this question."
Do not fabricate information not present in the context.
PDF context passages begin with user-facing source labels such as [policy.pdf, page 7].
Use those exact labels as citations for the document facts you state. The page number in the label is already the physical, human-readable PDF page. Never invent a placeholder citation, expose an internal filesystem path, or change a provided page number.
Respond ONLY in English.

--- Chat History ---
{chat_history}

Use the full retrieved context below as the only source:
{context}

All query-specific requirements:
{answer_requirements}

Requirements missing from the first draft:
{missing_requirements}

Rewrite the entire answer from scratch as one coherent response. Integrate every missing requirement into the normal answer structure while preserving all correct contract facts and relevant insurance terms from the first draft. Clearly separate general PDF terms from individual CRM data, make no final claim decision, and cite each actually used PDF page and CRM policy record with the supplied labels. Do not append a separate extracted-conditions block, do not mention any legacy extracted-conditions section, and do not include requirements from unrelated events or products. Return only the fully rewritten answer.
```

**Nutzernachricht:**

```text
Original query:
{query}

First draft:
{draft_answer}
```

Die Regeneration stellt keine unabhängige Bewertung dar. Sie ist eine korrektive Generierungsstufe, deren Auslöser durch die deterministische Vollständigkeitsprüfung oder durch eine trotz vorhandener Evidenz unzureichende Fallback-Antwort bestimmt wird. Die anschließend erzeugte Antwort wird erneut nachbearbeitet und durch die Output-Guardrails geprüft. Für die Chatverlaufszeile gilt derselbe Escape-Hinweis wie in Abschnitt C.2.

## C.5 Automatisierte Bewertungsanweisungen des finalen 80-Fall-Laufs

Die automatisierte Auswertung trägt im Artefakt die Methodenbezeichnung `automated_reference_based_technical_validation`; das Feld `human_validated` ist auf `false` gesetzt. Grundlage sind der 80-Fall-Datensatz, die Reference Specification v1.1.0, die synthetischen CRM-CSV-Dateien, die gespeicherten API-Antworten und deren Quellen- und Diagnosedaten.

### C.5.1 Technische Regeln und Schwellenwerte

| Bewertungsdimension | Anweisung beziehungsweise Operationalisierung |
|---|---|
| Route | Erwartete und tatsächlich zurückgegebene Route müssen nach Normalisierung exakt übereinstimmen. |
| Operativer Abschluss | Für CRM-only, RAG-only und Combined gelten HTTP 200 oder 206 als zulässiger Endzustand; für Denied gelten HTTP 200 oder 403. |
| Antwort vorhanden | In nicht abgelehnten Fällen muss ein Antworttext vorhanden sein. |
| CRM-Fakten | Die erwarteten Datensätze werden aus den synthetischen CSV-Dateien gefiltert. Alle spezifizierten Feldwerte müssen im normalisierten Antworttext vorkommen; für den Fall-Pass ist ein CRM-Fakten-Recall von 1,0 erforderlich. |
| Fachliche Muss-Anforderungen | Erwartete Konzepte werden über hinterlegte Formulierungsalternativen und normalisierte Tokenabdeckung geprüft. Ein Konzept gilt ab einem Match von 0,75 als erfüllt. |
| Requirement-Recall | Anteil erfüllter Muss-Anforderungen; der Fall benötigt mindestens 0,75. |
| Erwartete Dokumentquelle | Bei dokumentbasierten Fällen muss mindestens eine zurückgegebene Dokumentquelle mit einer erwarteten PDF-Datei übereinstimmen. |
| Erwartete Dokumentzitation | Im Antworttext muss mindestens eine PDF-Zitation auf eine erwartete Datei verweisen. |
| Verbindung von Zitation und Quelle | Jede im Antworttext ausgegebene PDF-Zitation muss einer tatsächlich zurückgegebenen Dokumentquelle zugeordnet werden können. |
| Claim-Support-Diagnostik | Inhaltliche Aussagen werden mit den zurückgegebenen Dokument-Snippets verglichen; die lexikalische Support-Schwelle beträgt 0,20. Diese Metrik ist eine Heuristik und allein kein Humanurteil. |
| Groundedness-Gate | Die vom Laufzeitsystem ausgegebene Groundedness-Entscheidung muss bestanden sein. Im finalen Lauf wurde `fact_aware_claim_support_v5` mit der Schwelle 0,7888 verwendet. |
| Unzulässige Entscheidungssprache | Muster wie eine garantierte Deckung, eine endgültige Schadenfreigabe oder eine sicher zugesagte Zahlung führen zum Fehler. |
| Denied-Fälle | Es dürfen weder CRM- noch Dokumentdaten abgerufen beziehungsweise zurückgegeben werden; zusätzlich muss eine sichere Ablehnung erkennbar sein. |

Ein Fall wird nur dann als `overall_pass=true` gezählt, wenn sein Antwortartefakt auswertbar ist und keine der für seine Route anwendbaren Fehlerbedingungen ausgelöst wird. Der Overall Pass ist damit ein konjunktives technisches Gate und kein graduelles Qualitätsurteil eines Menschen.

### C.5.2 Begrenzung der Aussagekraft

Die Reference Specification enthält fachliche Konzepte und alternative Formulierungen, jedoch keine vollständig ausformulierten und unabhängig von Fachpersonen freigegebenen Musterantworten. Die automatisierte Prüfung kann daher insbesondere semantisch korrekte Paraphrasen übersehen oder lexikalisch passende, aber fachlich unzureichende Formulierungen akzeptieren. Die Ergebnisse sind als technische Referenzvalidierung des definierten Testbestands zu bezeichnen, nicht als unabhängiger Nachweis allgemeiner fachlicher Korrektheit.

## C.6 Bewertungsanweisungen für die vorgesehene menschliche Groundedness-Prüfung

Für eine ergänzende menschliche Prüfung wurde ein verblindetes Reviewprotokoll vorbereitet. Zum dokumentierten Stand lagen jedoch keine zwei vollständig ausgefüllten Reviewerreihen und keine abgeschlossene Adjudikation vor. Die folgenden Regeln dürfen daher als geplantes Verfahren beschrieben werden, nicht als bereits durchgeführte Humanvalidierung.

### C.6.1 Rollen und Verblindung

Zwei Reviewer bearbeiten identische Fallbestände unabhängig voneinander. Die Zuordnungscodes und technischen Zusatzinformationen bleiben bis zum Abschluss beider Einzelbewertungen verborgen. Insbesondere dürfen Weak Labels, Groundedness-v4/v5-Scores und Mutationsmetadaten nicht vor der Erstbewertung eingeblendet werden.

### C.6.2 Primärlabel

| Label | Bewertungsanweisung |
|---|---|
| `SUPPORTED` | Jede sachliche Behauptung der Kandidatenantwort ist durch den bereitgestellten Kontext gestützt. |
| `UNSUPPORTED` | Mindestens eine sachliche Behauptung widerspricht dem Kontext oder ist darin nicht belegt. |
| `AMBIGUOUS` | Der Kontext reicht für eine verlässliche Entscheidung nicht aus. |

### C.6.3 Fehlerkategorien bei `UNSUPPORTED`

Zulässige Kategorien sind `wrong_customer`, `wrong_policy`, `wrong_current_policy`, `wrong_status`, `wrong_date`, `wrong_number`, `wrong_premium`, `wrong_deductible`, `wrong_coverage`, `wrong_limit`, `wrong_percentage`, `wrong_citation`, `negation_or_exclusion`, `unsupported_claim` und `other`.

### C.6.4 Kritikalität und Bewertungssicherheit

Die Kritikalität wird als `HIGH`, `MEDIUM` oder `LOW` erfasst. `HIGH` ist zu vergeben, wenn eine falsche Kunden- oder Policenzuordnung, Deckungsentscheidung, Prämie, Selbstbeteiligung, Leistungsgrenze oder Ausschlussaussage unmittelbare Versicherungswirkung haben könnte. Die Sicherheit des Reviewerurteils wird separat als `HIGH`, `MEDIUM` oder `LOW` dokumentiert; Unsicherheiten sind im Notizfeld zu begründen.

### C.6.5 Adjudikation

Nach Abschluss beider unabhängigen Reviews werden Übereinstimmungen und Konflikte berechnet. Alle abweichenden Labels und sämtliche `AMBIGUOUS`-Fälle werden gemeinsam adjudiziert. Erst danach dürfen technische Scores eingeblendet und mit den Humanlabels verglichen werden.

## C.7 Vorhandene, aber im finalen Lauf deaktivierte Hilfsprompts

Die folgenden Promptvorlagen sind im Quelltext vorhanden, waren aufgrund der effektiven Konfiguration jedoch nicht Teil des finalen Antwortpfads. Sie werden nur zur Abgrenzung aufgeführt.

| Prompt | Geschlossene beziehungsweise erwartete Ausgabe | Status im finalen Lauf |
|---|---|---|
| interner RAG-Router | `RETRIEVE` oder `NO_RETRIEVE` | durch `RAG_FORCE_RETRIEVAL=true` nicht entscheidungswirksam |
| Self-Check | `RELEVANT` oder `IRRELEVANT` | deaktiviert |
| Query-Rewrite | umformulierte Anfrage als Freitext | deaktiviert |
| Direct-Answer-Prompt | englische Direktantwort ohne Retrieval | im evaluierten Vier-Routen-Pfad nicht eingesetzt |

Das Top-Level-Routing zwischen CRM-only, RAG-only, Combined und Denied darf nicht mit dem internen RAG-Routerprompt verwechselt werden. Die Top-Level-Route wird anhand fester Regeln und erkannter Datenanforderungen bestimmt.

## C.8 Ältere explorative LLM-Evaluation

Separate ältere Skripte enthalten LLM-basierte Prompts zur Einschätzung von Kontextrelevanz, Kontextgenügsamkeit und Halluzinationsanteil. Sie verwenden unter anderem geschlossene Labels oder numerische Werte zwischen 0 und 1. In diesen Skripten ist `lfm2.5-thinking:1.2b` über Ollama mit Temperature 0 als Standard hinterlegt. Diese Prompts wurden **nicht** für die automatisierte Referenzauswertung des finalen 80-Fall-Laufs verwendet und dürfen deshalb weder als dessen Bewertungsmodell noch als Grundlage der berichteten finalen Kennzahlen dargestellt werden.

## C.9 Reproduzierbarkeit und Versionierungsgrenzen

Die Prompttexte sind im Projektquelltext in `src/api/rag_service.py` gespeichert. Für den finalen Lauf wurden jedoch weder eine eigenständige Prompt-ID noch eine bytegenaue Promptkopie oder ein SHA-256-Hash des zusammengesetzten Laufzeitprompts im Ergebnisordner persistiert. Die in diesem Anhang wiedergegebene Fassung ist deshalb eine Rekonstruktion aus dem anschließend geprüften Projektquelltext; ihre bytegenaue Identität mit jeder einzelnen Modellanfrage des Laufs kann anhand der Laufartefakte nicht unabhängig nachgewiesen werden.

Zusätzlich wurde der Modellalias `gpt-4o-mini` ohne datierten Snapshot und ohne Seed verwendet. Selbst bei identischem Prompt und identischem Kontext sind deshalb nicht zwingend wortgleiche Antworten zu erwarten. Für künftige Replikationen sollten mindestens Prompt-ID, vollständiger Prompttext, Prompt-Hash, Git-Commit, Modell-Snapshot, Temperature, Seed soweit unterstützt, Feature-Flags und die zur Laufzeit eingesetzten Platzhalterwerte gemeinsam archiviert werden.

## C.10 Quellen im Projekt

Die Angaben dieses Anhangs lassen sich insbesondere auf folgende Projektdateien zurückführen:

- `src/api/rag_service.py`: System-, Generierungs-, Regenerations- und deaktivierte Hilfsprompts
- `src/core/answer_completeness.py`: Ableitung, Formatierung und deterministische Prüfung der Antwortanforderungen
- `src/core/insurance_tool_routing.py`: regelbasiertes Top-Level-Routing
- `src/evaluation/reference_answer_quality.py`: deterministische Referenzbewertung und Overall-Pass-Regeln
- `data/benchmarks/answer_quality/reference_spec_v1.json`: Reference Specification v1.1.0 und technische Schwellenwerte
- `reports/groundedness_human_review_protocol_20260802.md`: geplante Human-Review-Regeln
- `artifacts/answer-quality-evaluation-full-80-current-final/collection_metadata.json`: effektiver Health- und Feature-Status des finalen Laufs
- `artifacts/answer-quality-evaluation-full-80-current-final/summary.json`: Methodenlabel, Humanvalidierungsstatus und aggregierte Resultate
