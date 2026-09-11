# Halluzinationsaudit des aktuellen Systems

## Geltungsbereich

Der Audit wurde am 10.09.2026 mit dem laufenden Backend und demselben PDF-RAG-only-
Benchmark mit 200 Fragen durchgefuehrt, der bereits fuer den historischen Vergleich
verwendet wurde. Das Backend meldete `claim_entailment_v1` als aktive
Groundedness-Implementierung. Die Ergebnisse gelten fuer den PDF-RAG-only-Pfad und
duerfen nicht ohne weitere Tests auf CRM-only, Combined oder Denied verallgemeinert
werden.

Der Audit ist automatisiert und nicht humanvalidiert. Als autoritative Evidenz wurden die
im Lauf zurueckgegebenen PDF-Seiten, die in der Antwort genannten Seiten und die
Referenzseiten des Benchmarks verwendet.

## Definitionen

- `supported`: Die Evidenz stuetzt den gesamten Claim mit Subjekt, Produkt, Zahlen,
  Bedingungen, Ausschluessen und Polaritaet.
- `contradicted`: Die Evidenz enthaelt eine explizit unvereinbare Aussage zum selben
  Gegenstand und Geltungsbereich.
- `insufficient_evidence`: Die Evidenz reicht fuer den Claim nicht aus. Das ist ein
  Halluzinations-Proxy, aber kein Beweis, dass die Aussage ausserhalb des Korpus falsch ist.
- `unknown`: Die automatische Entscheidung ist wegen Mehrdeutigkeit, Provenienz- oder
  Evaluatorunsicherheit offen.

Groundedness und Halluzination sind nicht identisch. Groundedness ist ein
Unterstuetzungsgrad. Halluzination ist hier ein Fehlerereignis auf Claim- oder
Antwortebene. Ein korpusintern unbelegter Claim kann wahr sein und bleibt dennoch fuer
diese geschlossene Versicherungsanwendung nicht freigabefaehig.

## Laufzeitergebnisse

| Ergebnis | Anzahl | Anteil an 200 |
|---|---:|---:|
| Nutzerseitig sichtbare faktische Antworten | 30 | 15,0 % |
| Post-Generation-Fallback | 116 | 58,0 % |
| Kontrollierte Enthaltung wegen unzureichender Information | 38 | 19,0 % |
| PII-Fallback | 1 | 0,5 % |
| Operativer Fehler/HTTP 504 | 15 | 7,5 % |

Die 116 Grounding-Fallbacks verteilten sich auf 58 Faelle mit mindestens einem
sensitiven, nicht ausreichend gestuetzten Claim, 53 Evaluatorfehler, vier Faelle mit
Reparaturbedarf/Provenienzunsicherheit und einen expliziten Widerspruch. Zusaetzlich
blieben zwei redigierte Antworten sichtbar, obwohl deren Groundedness-Diagnostik einen
Evaluatorfehler beziehungsweise einen sensitiven unsupported Claim meldete. Die
Priorisierung von `redact` gegenueber `fallback` ist daher eine offene Policy-Luecke.

## Halluzinationsmetriken

Von den 30 sichtbaren faktischen Antworten konnten 28 mit validierter atomarer
Claim-Extraktion und zwei mit einem vorsegmentierten Claim-Unit-Fallback ausgewertet
werden. Damit betraegt die Antwortabdeckung 100 %.

| Metrik | Ergebnis | Wilson-95-%-KI |
|---|---:|---:|
| Bestaetigte Widersprueche, Claim-Ebene | 0/57 = 0,00 % | 0,00-6,31 % |
| Bestaetigte Halluzination, Antwort-Ebene | 0/30 = 0,00 % | 0,00-11,35 % |
| Systemweite Exposition pro Anfrage | 0/200 = 0,00 % | 0,00-1,88 % |
| Automatischer Unsupported-Proxy, Claim-Ebene | 4/48 = 8,33 % | 3,29-19,55 % |
| Automatischer Unsupported-Proxy, Antwort-Ebene | 2/30 = 6,67 % | 1,85-21,32 % |
| Automatische Unknown-Rate, Antwort-Ebene | 7/30 = 23,33 % | 11,79-40,93 % |

Die vier Unsupported-Proxy-Claims gehoerten zu zwei Antworten (`pdf-rag-084` und
`pdf-rag-154`). Eine sekundaere Inspektion der gespeicherten Quellseiten zeigte, dass
beide Antworten durch die Seiten 10 beziehungsweise 29 gestuetzt waren. Diese Treffer
sind daher false positives des automatischen Auditors und keine bestaetigten
Halluzinationen. Diese Inspektion ersetzt keine unabhaengige fachliche Doppelannotation.

## Abgefangener Widerspruch

Im Fall `pdf-rag-124` fuegte die Verarbeitung vor dem Output Guardrail die Behauptung
hinzu, die Ersatzleistung bei Totalschaden oder Diebstahl sei auch Bestandteil der
Teilkaskoversicherung. Die Evidenz ordnete sie der Vollkaskoversicherung zu. Der Evaluator
klassifizierte den Claim als `contradicted`, begrenzte den Groundedness-Score auf 0,25
und ersetzte den gesamten Entwurf durch den sicheren Fallback. Im Test wurden damit
1/1 erkannte explizite Widersprueche vor der Nutzerexposition blockiert.

## Berechnung

Mit den Anzahlen `N_S`, `N_C`, `N_I` und `N_U` fuer supported, contradicted,
insufficient evidence und unknown gilt:

```text
H_claim_strict = N_C / (N_S + N_C + N_I + N_U)
H_claim_proxy  = (N_C + N_I) / (N_S + N_C + N_I)
U_claim        = N_U / (N_S + N_C + N_I + N_U)
H_response     = Antworten mit mindestens einem contradicted Claim / sichtbare faktische Antworten
H_exposure     = sichtbare Antworten mit mindestens einem contradicted Claim / alle Anfragen
```

Fuer jede Rate wurde das 95-%-Konfidenzintervall nach Wilson mit `z = 1,96`
berechnet. Die aktuelle Laufzeit berechnet ausserdem
`G = N_S / (N_S + N_C + N_I)`; unknown wird nicht in diesen Quotienten aufgenommen.
Bei mindestens einem bestaetigten Widerspruch wird `G` auf hoechstens 0,25 begrenzt.

## Systemreaktion

1. Query- und Context-Guardrails pruefen Injection, Secrets, PII und Entitaetsbindung.
2. CRM-only formatiert validierte CRM-Daten deterministisch ohne finale LLM-Generierung.
3. RAG-only und Combined erzeugen einen Entwurf aus gekennzeichneter PDF- und
   gegebenenfalls CRM-Evidenz.
4. `claim_entailment_v1` extrahiert Claims, auditiert die Extraktion und klassifiziert
   jeden Claim gegen die Evidenz. Zahlen, Waehrungen, Daten, Dauern, Orte, Produkte,
   Deckungsarten, Ausschluesse, Bedingungen, Polaritaet und Zitationsprovenienz werden
   separat kontrolliert.
5. Widerspruch, sensitiver unsupported Claim, Evaluatorfehler oder nicht aufloesbare
   Provenienzunsicherheit fuehren im aktuellen Pfad direkt zum Fallback; Quellen werden
   aus der finalen Ausgabe entfernt.
6. Eine begrenzte Claim-Reparatur existiert nur, wenn `low_groundedness` der einzige
   Blocker ist. Dann werden nur markierte schwache Claims umformuliert oder entfernt,
   erneut geprueft und gegebenenfalls durch eine deterministische Minimalantwort ersetzt.
   Im aktuellen 200-Fall-Lauf wurde dieser Reparaturpfad kein einziges Mal aktiviert.

## Einordnung

Der beobachtete strikte Wert von 0 % ist ein Punktschaetzer, keine Garantie. Das breite
Intervall von 0-11,35 % auf Antwortebene folgt aus nur 30 sichtbaren faktischen Antworten.
Die geringe Exposition wurde zudem mit einer hohen Post-Fallback-Rate von 58 % und einer
operativen Fehlerrate von 7,5 % erkauft. Fuer eine belastbare fachliche Rate sind eine
unabhaengige Annotation durch mindestens zwei fachkundige Personen, ein unangetasteter
Testsplit und getrennte Auswertungen fuer RAG-only, CRM-only und Combined erforderlich.

Der Konfigurationswert `0.7888` ist weiterhin aktiv, stammt aber aus einer historischen
Weak-Label-Kalibrierung des v5-Verfahrens. Er wurde fuer `claim_entailment_v1` nicht neu
kalibriert und darf deshalb nur als Deployment-Wert, nicht als wissenschaftlich
validierte aktuelle Halluzinationsschwelle bezeichnet werden. Die aktuelle Entscheidung
wird ausserdem bereits vor der Schwellenpruefung durch Claim-basierte Hard Gates bestimmt.

