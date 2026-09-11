# Prompt fuer GPT Prism: Halluzinationen in die Abschlussarbeit einarbeiten

Du bearbeitest die beigefuegte deutschsprachige Abschlussarbeit. Fuehre die folgenden
Aenderungen direkt im Dokument aus. Bewahre Stil, Terminologie, Nummerierung,
Querverweise, Tabellenlayout und Zitierstil der Arbeit. Veraendere keine fachlich
unbeteiligten Abschnitte. Erfinde keine Messwerte, Experimente, Humanlabels oder
Quellen. Verwende die unten angegebenen Zahlen exakt.

Setze beim Einfuegen alle deutschen Umlaute und das Eszett orthografisch korrekt. Die
ASCII-Schreibweise (`ae`, `oe`, `ue`, `ss`) in diesem Prompt dient nur der stabilen
Textuebertragung und soll nicht unveraendert in der Abschlussarbeit erscheinen.

## Zentrale Korrektur

Die Arbeit beschreibt an mehreren Stellen `fact_aware_claim_support_v5` als aktive
Laufzeitimplementierung. Das ist fuer den aktuellen Systemstand nicht mehr richtig. Aktiv
ist `claim_entailment_v1`. Der Wert `0,7888` steht weiterhin in
`config/groundedness_calibration.json`, ist aber nicht fuer `claim_entailment_v1` neu
kalibriert worden. Beschreibe ihn daher nur als historischen beziehungsweise operativen
Deployment-Wert. Stelle ihn nicht als humanvalidierte Halluzinationsschwelle und nicht als
alleiniges Freigabekriterium dar. Die aktuelle Policy verwendet Claim-basierte Hard Gates
fuer Widerspruch, sensitive unsupported Claims, Evaluatorfehler und Provenienzunsicherheit.

## 1. Kapitel 2 erweitern

Fuege direkt nach Abschnitt **2.5.2 Groundedness und Quellenangaben** einen neuen
Abschnitt **2.5.3 Halluzinationen und Abgrenzung zur Groundedness** ein. Nummeriere die
bisherigen Abschnitte 2.5.3 und 2.5.4 entsprechend zu 2.5.4 und 2.5.5 um. Verwende
folgenden Text:

### 2.5.3 Halluzinationen und Abgrenzung zur Groundedness

Der Begriff Halluzination wird in dieser Arbeit systemspezifisch operationalisiert. Als
Halluzination gilt eine ueberpruefbare faktische Aussage in einer generierten Antwort,
die durch die fuer die Anfrage autorisierte Evidenz nicht gestuetzt wird oder dieser
Evidenz widerspricht. Fuer die geschlossene Versicherungsanwendung bilden die
freigegebenen PDF-Dokumente und die eindeutig gebundenen CRM-Datensaetze die
massgebliche Evidenz. Allgemeines Modellwissen darf fehlende Vertrags- oder
Produktinformationen nicht ersetzen.

Halluzination und Groundedness sind nicht identisch. Groundedness beschreibt den Grad,
in dem eine Aussage aus dem bereitgestellten Kontext folgt. Halluzination bezeichnet
dagegen ein Fehlerereignis auf Claim- oder Antwortebene. Eine Aussage kann ausserhalb
des Systems sachlich wahr, innerhalb der konkreten Anfrage aber unbelegt sein. Umgekehrt
kann eine Aussage gegenueber einer fehlerhaften Quelle grounded sein, ohne objektiv wahr
zu sein. Die Arbeit trennt deshalb vier Relationen: `supported`, `contradicted`,
`insufficient_evidence` und `unknown`.

Als bestaetigte Halluzination wird in der strikten Auswertung nur ein Claim gezaehlt, der
einer autoritativen Quelle zum selben Gegenstand und Geltungsbereich explizit
widerspricht. Claims mit `insufficient_evidence` werden gemeinsam mit Widerspruechen als
konservativer Halluzinations-Proxy ausgewiesen. Sie sind jedoch kein Beweis fuer
faktische Falschheit. `unknown` bezeichnet offene automatische Entscheidungen und wird
separat berichtet. Diese Trennung ist erforderlich, weil automatisierte
Entailment-Verfahren sowohl false positives als auch false negatives erzeugen koennen.

Die Zerlegung einer Antwort in atomare Fakten folgt dem Grundgedanken von FActScore
[18]. Die getrennte Betrachtung von Quellenkorrektheit und Quellenabdeckung orientiert
sich an ALCE [19]. Beide Ansaetze begruenden die feingranulare Auswertung, ersetzen in
der Versicherungsdomaene aber keine unabhaengige fachliche Humanannotation.

## 2. Kapitel 3.8.4 vollstaendig ersetzen

Ersetze den gesamten bisherigen Abschnitt **3.8.4 Groundedness-Pruefung und
Fallback-Entscheidung**, einschliesslich Gleichung (3.2) und Algorithmus 4, durch den
folgenden aktualisierten Abschnitt. Entferne dabei die Aussage, das aktive Verfahren sei
deterministisch und verwende kein LLM.

### 3.8.4 Claim-basierte Groundedness- und Halluzinationspruefung

Die aktuelle Ausgabepruefung verwendet `claim_entailment_v1`. Das Verfahren kombiniert
strukturierte Modellurteile mit deterministischen Nachpruefungen. Zunaechst wird der
Antwortentwurf in semantische Einheiten und anschliessend in atomare Claims zerlegt. Eine
zweite strukturierte Modellabfrage auditiert, ob die Extraktion alle Aussagen, Zahlen,
Negationen, Bedingungen, Ausschluesse und Listenkontexte vollstaendig und unveraendert
enthaelt. Schlaegt dieser Audit auch nach einem begrenzten Reparaturversuch fehl, gilt die
Groundedness-Auswertung als technisch fehlgeschlagen.

Jeder Claim wird ausschliesslich gegen die bereitgestellte Dokument- und CRM-Evidenz
bewertet. Das Ergebnis ist eine der Relationen `supported`, `contradicted`,
`insufficient_evidence` oder `unknown`. `contradicted` setzt eine explizit unvereinbare
Evidenz zum selben Subjekt und Geltungsbereich voraus; fehlende Information wird als
`insufficient_evidence` und Mehrdeutigkeit als `unknown` behandelt. Fuer Zahlen,
Waehrungen, Prozentwerte, Daten, Dauern, Wartezeiten, Orte, Produktnamen,
Deckungsarten, Ausschluesse, Bedingungen und Polaritaet werden zusaetzliche strukturierte
Konsistenzpruefungen ausgefuehrt. Zitate muessen ausserdem auf eine tatsaechlich
ausgewaehlte Evidenzpassage zurueckfuehrbar sein.

Seien `N_S`, `N_C`, `N_I` und `N_U` die Anzahlen der supported, contradicted,
insufficient-evidence und unknown Claims. Der operative Groundedness-Score lautet

`G = N_S / (N_S + N_C + N_I)`,

sofern der Nenner groesser als null ist. Unknown Claims werden aus diesem Quotienten
ausgeschlossen und separat ausgewiesen. Bei mindestens einem bestaetigten Widerspruch
wird der Score auf hoechstens 0,25 begrenzt. Der in der Laufzeitkonfiguration enthaltene
Wert `tau = 0,7888` wird erst nach den Claim-basierten Hard Gates geprueft. Da dieser Wert
aus der historischen Weak-Label-Kalibrierung von
`fact_aware_claim_support_v5` stammt und fuer `claim_entailment_v1` nicht neu kalibriert
wurde, ist er ein Deployment-Wert und kein aktueller wissenschaftlicher
Kalibrierungsnachweis.

Die Freigabelogik arbeitet fail-closed auf Claim-Ebene. Ein bestaetigter Widerspruch, ein
sensitiver unbelegter Claim, ein Evaluatorfehler oder eine nicht aufgeloeste
Provenienzunsicherheit fuehren zur sicheren Ersatzantwort. Ein rein scorebasierter
`low_groundedness`-Fehler kann einen begrenzten Reparaturpfad ausloesen: Bereits
gestuetzte Claims bleiben erhalten; nur explizit markierte schwache Claims werden anhand
der vorhandenen Evidenz umformuliert oder entfernt. Danach wird die vollstaendige
Antwort erneut geprueft. Besteht sie weiterhin nicht, wird eine deterministische
Minimalantwort versucht; andernfalls wird der gesamte Entwurf durch den Fallback ersetzt
und die Quellenliste geleert.

Verwende fuer Algorithmus 4 folgenden Ablauf:

```text
Algorithmus 4: Claim-basierte Pruefung und sichere Ausgabe
Eingabe: Antwortentwurf A, autorisierte Evidenz D, Anfrage q, Deployment-Wert tau
1: Zerlege A in Antwortsegmente und atomare Claims.
2: Auditiere Vollstaendigkeit und Treue der Claim-Extraktion.
3: Wenn der Extraktionsaudit nach einem Reparaturversuch fehlschlaegt: gib Fallback zurueck.
4: Klassifiziere jeden Claim gegen D als supported, contradicted,
   insufficient_evidence oder unknown.
5: Pruefe sensitive Werte, Bedingungen, Ausschluesse, Polaritaet,
   Entitaetsbindung und Zitationsprovenienz.
6: Wenn ein Widerspruch, sensitiver unsupported Claim oder nicht aufloesbarer
   Evaluator-/Provenienzfehler vorliegt: gib Fallback zurueck.
7: Berechne G = N_S / (N_S + N_C + N_I); begrenze G bei Widerspruch auf 0,25.
8: Wenn G < tau der einzige Blocker ist: repariere nur markierte schwache Claims.
9: Pruefe die reparierte Antwort erneut.
10: Wenn sie nicht besteht: versuche eine evidenzbasierte Minimalantwort.
11: Wenn auch diese nicht besteht: gib Fallback ohne Quellen zurueck.
12: Andernfalls: gib die gepruefte Antwort mit validierten Quellen zurueck.
```

### 3.8.5 Operationalisierung der Halluzinationsrate

Fuege direkt nach 3.8.4 den folgenden neuen Unterabschnitt ein:

Die Halluzinationsrate wird getrennt auf Claim-, Antwort- und Systemebene berechnet. Die
strikte Claim-Rate, der konservative Unsupported-Proxy und die Unknown-Rate lauten:

`H_claim_strict = N_C / (N_S + N_C + N_I + N_U)`,

`H_claim_proxy = (N_C + N_I) / (N_S + N_C + N_I)`,

`U_claim = N_U / (N_S + N_C + N_I + N_U)`.

Auf Antwortebene ist `H_response_strict` der Anteil der sichtbaren faktischen Antworten,
die mindestens einen contradicted Claim enthalten. Die systemweite Expositionsrate
`H_exposure_strict` verwendet dagegen alle Anfragen als Nenner. Fallbacks, kontrollierte
Enthaltungen und technische Fehler enthalten keine sichtbaren faktischen Claims und
werden deshalb aus der Claim- und Antwortmetrik ausgeschlossen, bleiben aber im Nenner
der systemweiten Expositionsrate. Alle Anteilswerte werden zusammen mit einem
Wilson-95-%-Konfidenzintervall berichtet. Fuer beobachteten Anteil `p = k/n` und
`z = 1,96` gilt:

`KI_W = (p + z^2/(2n) +/- z*sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n)`.

## 3. Kapitel 4.3.5 aktualisieren

Behalte die bestehenden Beschreibungen von CRM-/Dokumentherkunft, OpenAI Responses API,
`gpt-4o-mini`, `store=false`, PII-Pruefung und Logging bei. Ersetze jedoch alle Saetze,
die `fact_aware_claim_support_v5` als aktive Groundedness-Implementierung bezeichnen,
durch folgenden Text:

Die aktive Groundedness- und Halluzinationspruefung ist in
`src/core/claim_groundedness.py` als `claim_entailment_v1` implementiert. Sie verwendet
strukturierte Modellantworten fuer Claim-Extraktion, Extraktionsaudit und
Entailment-Klassifikation. Deterministische Nachpruefungen validieren sensitive Fakten,
Deckungspolaritaet, Subjekt- und Geltungsbereich sowie die Provenienz der angefuehrten
Evidenzzitate. Das Output Guardrail in
`src/guardrails/integrations/nemo_actions.py` setzt daraus Hard Gates fuer
Evaluatorfehler, Widerspruch, sensitive unsupported Claims und nicht aufgeloeste
Unsicherheit um. Die Orchestrierung in `src/api/rag_service.py` ersetzt einen nicht
freigegebenen Entwurf durch eine sichere Ersatzantwort und entfernt bei Block oder
Fallback die Quellen aus der finalen Ausgabe.

Eine gezielte Antwortreparatur ist implementiert, wird aber nur ausgefuehrt, wenn
`low_groundedness` der einzige Blocker ist. Sie darf gestuetzte Claims nicht veraendern
und nur diagnostizierte schwache Claims umformulieren oder entfernen. Anschliessend wird
die reparierte Antwort erneut durch dasselbe Guardrail geprueft. Im 200-Fall-Lauf vom
10.09.2026 wurde dieser Reparaturpfad nicht aktiviert; alle 116 Grounding-bedingten
Post-Fallbacks wurden direkt unterdrueckt. Die aktuelle Wirkung ist daher vor allem
Expositionsvermeidung durch Abstention/Fallback und nicht nachtraegliche Korrektur jeder
problematischen Antwort.

Ergaenze als Implementierungsgrenze:

Bei zwei Antworten blieb eine PII-Redaktion als finale Aktion bestehen, obwohl die
Groundedness-Diagnostik einen Evaluatorfehler beziehungsweise einen sensitiven
unsupported Claim meldete. Die Aktionsprioritaet zwischen `redact` und `fallback` muss
daher in einer weiteren Revision explizit geschlossen und regressionsgetestet werden.

## 4. Ergebnisse in Kapitel 5 einfuegen

Fuege nach dem bisherigen Abschnitt **5.3.4 Groundedness- und Zitationsergebnisse** einen
neuen Abschnitt **5.3.5 Halluzinations- und Expositionsaudit des aktuellen Systems** ein.
Nummeriere den bisherigen Abschnitt 5.3.5 und alle folgenden Unterabschnitte innerhalb
von 5.3 entsprechend weiter. Verwende folgenden Text und die Tabelle:

### 5.3.5 Halluzinations- und Expositionsaudit des aktuellen Systems

Am 10.09.2026 wurde der PDF-RAG-only-Benchmark mit 200 unveraenderten Fragen erneut
gegen das laufende Backend ausgefuehrt. Der Lauf verwendete `gpt-4o-mini` fuer die
Antwortgenerierung und `claim_entailment_v1` als aktives Output Guardrail. Die Auswertung
ist automatisiert und nicht humanvalidiert. Sie gilt ausschliesslich fuer den
PDF-RAG-only-Pfad.

Von 200 Anfragen lieferten 185 einen HTTP-200-Abschluss; 15 Anfragen endeten mit HTTP
504. Nur 30 Antworten enthielten fuer den Nutzer sichtbare faktische Inhalte. In 116
Faellen ersetzte das Post-Generation-Guardrail den Entwurf durch einen sicheren
Fallback, 38 Antworten enthielten eine kontrollierte Enthaltung wegen unzureichender
Information, und eine Antwort wurde wegen personenbezogener beziehungsweise sensitiver
Informationen durch einen PII-Fallback ersetzt.

| Kennwert | Ergebnis | Wilson-95-%-Konfidenzintervall |
|---|---:|---:|
| Operational Completion | 185/200 = 92,50 % | 87,996-95,403 % |
| Sichtbare faktische Antworten | 30/200 = 15,00 % | 10,71-20,61 % |
| Post-Generation-Fallback | 116/200 = 58,00 % | 51,07-64,63 % |
| Kontrollierte Enthaltung | 38/200 = 19,00 % | 14,17-25,00 % |
| Operative Fehler/HTTP 504 | 15/200 = 7,50 % | 4,60-12,00 % |
| Bestaetigte Widersprueche in sichtbaren Claims | 0/57 = 0,00 % | 0,00-6,31 % |
| Strikte Halluzinationsrate sichtbarer Antworten | 0/30 = 0,00 % | 0,00-11,35 % |
| Systemweite strikte Halluzinationsexposition | 0/200 = 0,00 % | 0,00-1,88 % |
| Automatischer Unsupported-Proxy auf Antwortebene | 2/30 = 6,67 % | 1,85-21,32 % |
| Unknown auf Antwortebene | 7/30 = 23,33 % | 11,79-40,93 % |

Die 30 sichtbaren faktischen Antworten wurden vollstaendig nachbewertet. Fuer 28
Antworten lag eine validierte atomare Extraktion mit insgesamt 57 Claims vor; zwei
listenhafte Antworten wurden ersatzweise auf vorsegmentierten Claim Units bewertet. Der
automatische Auditor markierte vier von 48 entschiedenen atomaren Claims in zwei
Antworten als `insufficient_evidence`. Die anschliessende sekundaere Quelleninspektion
zeigte jedoch, dass die betreffenden Aussagen auf den gespeicherten Seiten 10 und 29
belegt waren. Diese Treffer werden deshalb als false positives des Auditors und nicht als
bestaetigte Halluzinationen eingeordnet. Eine unabhaengige fachliche Doppelannotation
liegt weiterhin nicht vor.

Vor der Ausgabe wurde genau ein expliziter Widerspruch erkannt. Im Fall `pdf-rag-124`
behauptete ein nachgelagerter Antwortschritt, die Ersatzleistung bei Totalschaden oder
Diebstahl gehoere auch zur Teilkaskoversicherung. Die Evidenz ordnete diese Leistung der
Vollkaskoversicherung zu. `claim_entailment_v1` klassifizierte den Claim als
`contradicted`, begrenzte den Score auf 0,25 und ersetzte den Entwurf durch den sicheren
Fallback. Damit wurde im beobachteten Lauf 1/1 erkannter expliziter Widersprueche vor
der Nutzerexposition blockiert.

Die 116 Grounding-bedingten Post-Fallbacks bestanden aus 58 Faellen mit sensitiven
unsupported Claims, 53 Evaluatorfehlern, vier Faellen mit Reparaturbedarf oder
Provenienzunsicherheit und einem expliziten Widerspruch. Die niedrige beobachtete
Expositionsrate ist daher gemeinsam mit der hohen Fallback-Rate und der mittleren
Latenz von 119,44 Sekunden zu interpretieren. Das System reduziert die sichtbare
Halluzinationsgefahr in diesem Lauf vor allem durch konservative Unterdrueckung, nicht
durch eine hohe Antwortabdeckung.

## 5. Diskussion in Abschnitt 6.4.2 ergaenzen

Ergaenze am Ende von **6.4.2 Antwortqualitaet und Groundedness** folgende Absaetze:

Der zusaetzliche Halluzinationsaudit bestaetigt die begriffliche Trennung von
Groundedness und Halluzination. Unter den 30 sichtbaren faktischen Antworten wurde kein
expliziter Widerspruch bestaetigt. Der Punktschaetzer der strikten Halluzinationsrate
betraegt damit 0 %, das Wilson-95-%-Konfidenzintervall reicht wegen der kleinen Anzahl
sichtbarer Antworten jedoch bis 11,35 %. Der Wert darf daher nicht als Nachweis einer
halluzinationsfreien Anwendung interpretiert werden. Auch die systemweite beobachtete
Exposition von 0/200 besitzt mit 0-1,88 % ein nichttriviales Unsicherheitsintervall.

Die Sicherheitswirkung steht einem deutlichen Utility-Verlust gegenueber. 58 % aller
Anfragen endeten nach der Generierung im Fallback, weitere 19 % in einer kontrollierten
Enthaltung und 7,5 % in einem operativen Fehler. Damit zeigt der Lauf nicht nur eine
niedrige sichtbare Widerspruchsrate, sondern auch ein stark konservatives System mit
geringer faktischer Antwortabdeckung von 15 %. Fuer den produktiven Einsatz muessen
Halluzinationsschutz, Antwortabdeckung, Evaluatorrobustheit und Latenz gemeinsam
optimiert werden.

Die automatische Unsupported-Rate erwies sich zudem als empfindlich gegen
Evidenzauswahl und Zitationsprovenienz. Zwei automatisch markierte Antworten waren bei
sekundaerer Seiteninspektion belegt. Zukuenftige Evaluationen sollten deshalb mindestens
zwei unabhaengige fachkundige Annotierende, ein dokumentiertes
Konfliktaufloesungsverfahren, einen unangetasteten Testsplit sowie getrennte Benchmarks
fuer RAG-only, CRM-only und Combined verwenden. Erst damit laesst sich eine fachlich
belastbare Halluzinationsrate mit Inter-Annotator-Uebereinstimmung angeben.

## 6. Anhang D.1 ersetzen

Ersetze **D.1 Groundedness- und Guardrail-Konfiguration** vollstaendig. Entferne die
alten v5-Formeln mit `L_ij`, `F_ij`, `U_ij`, `G_0`, die Cap-Tabelle D.1 und den beschriebenen
v4-Fallback. Verwende stattdessen eine kompakte technische Fassung der neuen Methodik:

### D.1 Claim-Entailment- und Guardrail-Konfiguration

Die aktive Implementierung traegt die Versionskennung `claim_entailment_v1`. Nach dem
Entfernen reiner Quellenzeilen wird der Antwortentwurf strukturerhaltend in Saetze,
Listenpunkte und abhaengige Listenueberschriften zerlegt. Ein strukturiertes Modell erzeugt
atomare Claims mit Verweisen auf die urspruenglichen Einheiten. Deterministische
Validierungen verlangen, dass alle erforderlichen Einheiten und sensitiven Fakten erhalten
bleiben. Ein zweiter strukturierter Modellaufruf auditiert die Treue und Vollstaendigkeit
der Extraktion. Pro Phase ist hoechstens ein Reparaturversuch vorgesehen.

Fuer jeden Claim werden begrenzte Evidenzfenster ausgewaehlt und die Relation
`supported`, `contradicted`, `insufficient_evidence` oder `unknown` bestimmt. Ein
Supported- oder Contradicted-Urteil wird nur akzeptiert, wenn die Evidenzzitate auf die
ausgewaehlten Fenster zurueckgefuehrt werden koennen, der Subjekt- und Geltungsbereich
uebereinstimmt und die Modellkonfidenz mindestens 0,85 betraegt. Deterministische
Nachpruefungen erfassen Identifikatoren, Zahlen, Geldbetraege, Prozentwerte, Daten,
Dauern, Orte, Deckungsarten und Polaritaet. Fehlende oder nicht verifizierbare Zitate
koennen ein Urteil zu `unknown` oder `insufficient_evidence` herabstufen.

Der Score wird als `G = N_S/(N_S+N_C+N_I)` berechnet; `unknown` wird separat
ausgewiesen. Enthalten die Claims einen bestaetigten Widerspruch, gilt zusaetzlich
`G = min(G, 0,25)`. Die Konfiguration enthaelt `tau = 0,7888`, dessen Herkunft jedoch
zum historischen v5-Weak-Label-Verfahren gehoert. Fuer die aktuelle Implementierung ist
keine neue Human- oder Hold-out-Kalibrierung nachgewiesen.

Die Laufzeitentscheidung verwendet vor dem numerischen Vergleich folgende Hard Gates:

| Diagnose | Aktuelle Reaktion |
|---|---|
| Evaluatorstatus `failed` | Fallback |
| Mindestens ein `contradicted` Claim | Fallback |
| Mindestens ein sensitiver `unknown` oder `insufficient_evidence` Claim | Fallback |
| Nicht-sensitiver Unknown- oder Provenienzfehler | Fallback/Reparaturbedarf |
| Nur `low_groundedness` bei `G < tau` | begrenzter Reparaturversuch, erneute Pruefung, sonst Fallback |

Kontrollierte Enthaltungen ohne faktische Claims werden als nicht anwendbar fuer die
Claim-Metrik behandelt. Bei Fallback oder Block wird die Quellenliste der finalen Antwort
geleert. Die Auditdiagnostik speichert Algorithmusversion, Relation Counts,
Claim-Details, Extraktions- und Urteilsversuche, Fehlerstufe und Entscheidungspfad.

## 7. Literatur und Konsistenzpruefung

Nutze die bereits vorhandenen Referenzen zu FActScore [18] und ALCE [19]. Pruefe, ob die
Arbeit bereits eine allgemeine Halluzinationsquelle enthaelt; falls nicht, ergaenze Ji et
al. zur begrifflichen Einordnung. Ergaenze fuer das Konfidenzintervall Wilson (1927).
Passe die laufenden Literaturzahlen automatisch an, falls der Zitierstil numerisch ist.

Zu verwendende Quellen:

- Min et al. (2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in
  Long Form Text Generation*, EMNLP, DOI 10.18653/v1/2023.emnlp-main.741.
- Gao et al. (2023), *Enabling Large Language Models to Generate Text with Citations*,
  EMNLP, DOI 10.18653/v1/2023.emnlp-main.398.
- Ji et al. (2023), *Survey of Hallucination in Natural Language Generation*, ACM
  Computing Surveys, DOI 10.1145/3571730.
- Wilson (1927), *Probable Inference, the Law of Succession, and Statistical Inference*,
  Journal of the American Statistical Association 22(158), 209-212,
  DOI 10.1080/01621459.1927.10502953.

Fuehre abschliessend eine globale Konsistenzsuche nach
`fact_aware_claim_support_v5`, `v5`, `Gleichung (3.2)`, `Algorithmus 4`, `0,7888`,
`deterministisch`, `56/60`, `93,33 %`, `Groundedness-Mittelwert` und
`SAFETY_FAIL_CLOSED=false` durch. Historische Ergebnisse duerfen erhalten bleiben, wenn
sie eindeutig als historisch und nicht als aktueller Systemstand markiert sind. Jede
Beschreibung des aktiven Systems muss dagegen `claim_entailment_v1` und den Lauf vom
10.09.2026 verwenden. Aktualisiere Inhalts-, Tabellen- und Algorithmenverzeichnis sowie
alle Querverweise und Seitenzahlen. Gib am Ende eine kurze Aenderungsliste aus.
