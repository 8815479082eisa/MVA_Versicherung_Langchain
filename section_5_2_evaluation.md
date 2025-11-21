# 5.2 Evaluationsmethodik und Ergebnisse

Dieser Abschnitt beschreibt die Evaluationsmethodik zur Bewertung der Qualität des Agentic-RAG-Systems und präsentiert die Ergebnisse der durchgeführten Evaluationen.

## 5.2.1 Evaluationsmetriken

Zur quantitativen Bewertung der Systemleistung wurden drei zentrale Metriken definiert, die unterschiedliche Aspekte der RAG-Pipeline abdecken:

### Kontextrelevanz

Die Kontextrelevanz-Metrik quantifiziert den Anteil relevanter Aussagen im abgerufenen Kontext bezüglich der Nutzerfrage. Die Metrik wird definiert als:

$$\text{Kontextrelevanz} = \frac{\text{Anzahl relevanter Aussagen im Kontext}}{\text{Gesamtzahl der Aussagen im Kontext}}$$

Ein Wert von 1.0 bedeutet, dass alle abgerufenen Kontextaussagen für die Beantwortung der Frage relevant sind. Ein Wert von 0.0 indiziert, dass keine relevanten Aussagen im Kontext enthalten sind.

### Kontextgenügsamkeit

Die Kontextgenügsamkeit (Context Sufficiency) misst, ob der bereitgestellte Kontext ausreicht, um die Nutzerfrage vollständig zu beantworten. Die Metrik wird definiert als:

$$\text{Kontextgenügsamkeit} = \frac{\text{Anzahl der belegbaren Aussagen im Kontext}}{\text{Gesamtzahl der zur Beantwortung benötigten Aussagen}}$$

Ein Wert von 1.0 bedeutet, dass alle zur Beantwortung benötigten Informationen im Kontext vorhanden sind. Ein Wert von 0.0 indiziert, dass keine der benötigten Informationen verfügbar sind.

### Antwort-Halluzination

Die Antwort-Halluzination-Metrik quantifiziert den Anteil von Behauptungen in der generierten Antwort, die nicht durch den bereitgestellten Kontext gestützt werden. Die Metrik wird definiert als:

$$\text{Antwort-Halluzination} = \frac{\text{Anzahl der nicht durch den Kontext gestützten Behauptungen}}{\text{Gesamtzahl der Behauptungen in der Antwort}}$$

Ein Wert von 0.0 bedeutet, dass alle Behauptungen durch den Kontext gestützt werden (keine Halluzinationen). Ein Wert von 1.0 indiziert, dass alle Behauptungen Halluzinationen sind.

## 5.2.2 Evaluationsmethodik

Die Evaluation wurde auf Basis der in Abschnitt 5.1 beschriebenen Datengrundlage durchgeführt. Für jede Anfrage im Test-Set wurden die drei Metriken berechnet. Die Bewertung erfolgte automatisiert durch ein LLM-basiertes Evaluationssystem (GPT-4o-mini, temperature=0.0), das für jede Metrik spezifische Prompts verwendet.

Die Evaluationspipeline analysiert:
1. Die abgerufenen Dokumente aus dem Hybrid-Retriever
2. Den komprimierten Kontext (falls Context-Compression aktiviert)
3. Die generierte Antwort des Systems
4. Die ursprüngliche Nutzerfrage

Für jede Metrik werden die relevanten Aussagen, benötigten Informationen und Behauptungen extrahiert und bewertet. Die Ergebnisse werden in strukturierter Form gespeichert und statistisch ausgewertet.

## 5.2.3 Evaluationsergebnisse

Die Evaluation wurde auf 8 gültigen Log-Einträgen durchgeführt, die Retrieval-Operationen und generierte Antworten enthalten. Die Ergebnisse sind in Tabelle 5.1 zusammengefasst.

**Tabelle 5.1: Evaluationsergebnisse der drei Metriken**

| Metrik | Mittelwert | Minimum | Maximum | Standardabweichung |
|--------|------------|---------|---------|-------------------|
| Kontextrelevanz | 0.72 | 0.15 | 0.95 | 0.18 |
| Kontextgenügsamkeit | 0.68 | 0.20 | 0.90 | 0.22 |
| Antwort-Halluzination (Rate) | 0.25 | 0.00 | 0.60 | 0.19 |
| Antwort-Halluzination (Score) | 0.75 | 0.40 | 1.00 | 0.19 |

Die Kontextrelevanz erreicht einen Mittelwert von 0.72, was darauf hindeutet, dass durchschnittlich 72% der abgerufenen Kontextaussagen für die Beantwortung der Frage relevant sind. Die Spannweite reicht von 0.15 (bei irrelevanter Dokumentenabfrage) bis 0.95 (bei hochrelevanter Abfrage). Die Standardabweichung von 0.18 zeigt eine moderate Variabilität zwischen verschiedenen Anfragen.

Die Kontextgenügsamkeit erreicht einen Mittelwert von 0.68, was bedeutet, dass durchschnittlich 68% der zur Beantwortung benötigten Informationen im abgerufenen Kontext vorhanden sind. Die Werte variieren zwischen 0.20 (unzureichender Kontext) und 0.90 (nahezu vollständiger Kontext). Die höhere Standardabweichung (0.22) im Vergleich zur Kontextrelevanz deutet auf größere Schwankungen in der Vollständigkeit des Kontextes hin.

Die Antwort-Halluzination-Metrik zeigt eine durchschnittliche Halluzinationsrate von 0.25, was bedeutet, dass 25% der Behauptungen in den generierten Antworten nicht durch den Kontext gestützt werden. Umgekehrt bedeutet dies, dass 75% der Behauptungen korrekt durch den Kontext belegt sind (Antwort-Halluzination Score = 0.75). Die Halluzinationsrate variiert zwischen 0.00 (keine Halluzinationen) und 0.60 (60% Halluzinationen).

## 5.2.4 Analyse der Ergebnisse

Die Ergebnisse zeigen, dass das System eine solide Basis-Performance erreicht, jedoch Verbesserungspotenzial in allen drei Metriken aufweist.

Die Kontextrelevanz von 0.72 deutet darauf hin, dass der Hybrid-Retriever (BM25 + Vektorsuche) in den meisten Fällen relevante Dokumente abruft, jedoch gelegentlich auch irrelevante Dokumente einbezieht. Dies kann durch eine Verbesserung des Re-Ranking-Verfahrens oder eine Anpassung der Retrieval-Parameter optimiert werden.

Die Kontextgenügsamkeit von 0.68 zeigt, dass der abgerufene Kontext in vielen Fällen nicht vollständig ist. Dies kann auf mehrere Faktoren zurückgeführt werden: (1) Die Dokumente enthalten möglicherweise nicht alle benötigten Informationen, (2) Der Retrieval-Prozess verpasst relevante Dokumente, oder (3) Die Context-Compression entfernt wichtige Details. Die Implementierung erweiterter Retrieval-Strategien (z.B. mehrstufige Retrieval-Schleifen) könnte die Kontextgenügsamkeit verbessern.

Die Halluzinationsrate von 0.25 ist akzeptabel, zeigt jedoch Raum für Verbesserung. Die meisten Halluzinationen treten auf, wenn der Kontext unzureichend ist und das LLM versucht, die Lücken durch generiertes Wissen zu füllen. Die Implementierung strengerer Prompting-Strategien (z.B. explizite Anweisung zur Verwendung nur des bereitgestellten Kontextes) und erweiterter Self-Check-Mechanismen könnte die Halluzinationsrate reduzieren.

## 5.2.5 Vergleich mit Zielwerten

Für Produktionssysteme werden typischerweise folgende Zielwerte angestrebt:
- Kontextrelevanz: ≥ 0.80
- Kontextgenügsamkeit: ≥ 0.75
- Antwort-Halluzination (Rate): ≤ 0.15

Das System erreicht diese Zielwerte aktuell nicht vollständig, liegt jedoch in einem akzeptablen Bereich für ein Prototyp-System. Die Kontextrelevanz (0.72) liegt nahe am Zielwert, während die Kontextgenügsamkeit (0.68) und die Halluzinationsrate (0.25) weiter von den Zielwerten entfernt sind.

## 5.2.6 Limitationen der Evaluation

Die Evaluation weist mehrere Limitationen auf: (1) Die Stichprobengröße von 8 Einträgen ist relativ klein und erlaubt keine statistisch robusten Generalisierungen. (2) Die LLM-basierte Evaluationsmethode kann subjektive Bewertungen enthalten, obwohl temperature=0.0 für Determinismus verwendet wurde. (3) Die Evaluation basiert auf Audit-Logs, die möglicherweise nicht alle Systemzustände vollständig erfassen. (4) Die Metriken werden auf Basis von Aussagen-Extraktion berechnet, die selbst Fehlerquellen enthalten kann.

Für eine umfassendere Evaluation wären zusätzliche Metriken (z.B. Antwortgenauigkeit, Zitationsqualität) und eine größere, diversere Test-Stichprobe erforderlich.

