# 3.8 Algorithmen und Pseudocode

Die zentralen Abläufe des Prototyps werden nachfolgend in abstrahierter Form beschrieben. Die Darstellung orientiert sich an der implementierten Verarbeitung, verzichtet jedoch auf technische Nebenbedingungen wie Logging und Laufzeitmessung. Das Routing erfolgt deterministisch und regelbasiert. Dadurch bleibt die Wahl der Datenquellen nachvollziehbar und ist nicht von einer zusätzlichen LLM-Entscheidung abhängig.

## 3.8.1 Routing und Auswahl des Verarbeitungspfads

Für eine Anfrage \(q\) werden drei binäre Merkmale gebildet: \(I(q)\) bezeichnet das Vorhandensein einer begrenzenden Kunden- oder Vertragsreferenz, beispielsweise Name, E-Mail-Adresse, Policen- oder Schadennummer. \(C(q)\) kennzeichnet einen Bezug zu CRM-Daten und \(D(q)\) einen Bedarf an Dokumentwissen, etwa zu Deckung, Bedingungen oder Ausschlüssen. Vor dieser Einordnung werden unzulässige Sammelabfragen sowie sicherheitskritische Eingaben gesperrt.

\[
R(q)=
\begin{cases}
\text{Denied}, & \text{wenn eine Sicherheits- oder Bereichsverletzung vorliegt},\\
\text{RAG-only}, & \neg\bigl(I(q) \land C(q)\bigr),\\
\text{Combined}, & I(q) \land C(q) \land D(q),\\
\text{CRM-only}, & I(q) \land C(q) \land \neg D(q).
\end{cases}
\]

**Tabelle 3.x: Entscheidungskriterien des Routings**

| Erkannter Informationsbedarf | Gewählter Pfad | Verwendete Evidenz |
|---|---|---|
| Allgemeines Versicherungswissen ohne benötigten Kundendatensatz | RAG-only | Versicherungsdokumente |
| Kunden-, Policen- oder Schadenfakten ohne Dokumentfrage | CRM-only | ausgewählte CRM-Datensätze |
| Individuelle Vertragsdaten und allgemeine Bedingungen | Combined CRM–RAG | CRM-Datensätze und Dokumentpassagen |
| Unzulässige Bereichsabfrage oder harter Sicherheitsverstoß | Denied | keine fachliche Verarbeitung |

**Algorithmus 1: Deterministische Pfadauswahl**

```text
Eingabe: Anfrage q
Ausgabe: Verarbeitungsmodus R

1  wenn q ein hartes Sicherheitssignal enthält: gib DENIED zurück
2  wenn q eine unzulässige CRM-Sammelabfrage verlangt: gib DENIED zurück
3  I ← enthält q eine begrenzende Kunden-, Policen- oder Schadensreferenz?
4  C ← enthält q einen CRM-bezogenen Informationsbedarf?
5  D ← enthält q einen dokumentbezogenen Informationsbedarf?
6  wenn nicht (I und C): gib RAG_ONLY zurück
7  wenn D: gib COMBINED zurück
8  gib CRM_ONLY zurück
```

## 3.8.2 Hybrid Retrieval und Reranking

Im Dokumentenpfad werden eine BM25-basierte Keyword-Suche und eine Embedding-basierte Vektorsuche parallel auf demselben Chunk-Bestand ausgeführt. Die aktuellen Modelle sind `BAAI/bge-m3` für die Embeddings und `cross-encoder/ms-marco-MiniLM-L-6-v2` für das Reranking. Die Ranglisten werden mittels Reciprocal Rank Fusion (RRF) zusammengeführt. Für ein Dokumentsegment \(d\) ergibt sich mit der Konstante \(k_{RRF}=60\):

\[
s_{RRF}(d)=
\sum_{m\in\{BM25,Vektor\}}
\frac{\mathbb{1}[d\in L_m]}{k_{RRF}+r_m(d)}.
\]

Dabei bezeichnet \(r_m(d)\) den Rang von \(d\) in der Ergebnisliste \(L_m\). Eine balancierte Kandidatenauswahl stellt sicher, dass semantische und lexikalische Treffer in die fusionierte Menge eingehen. Anschließend können bis zu drei inhaltlich passende Nachbar-Chunks derselben Quelle ergänzt werden. Der Cross-Encoder bewertet danach Anfrage und Chunk gemeinsam. Neben diesem Score berücksichtigt die finale Sortierung deterministische Signale zur Produktzugehörigkeit und zur fachlichen Evidenzpriorität. In der aktuellen Konfiguration werden acht fusionierte Kandidaten abgerufen und höchstens fünf Chunks an die Antwortgenerierung übergeben.

**Algorithmus 2: Hybrid Retrieval mit Reranking**

```text
Eingabe: Dokumentanfrage q, Zielgrößen K und K_R
Ausgabe: gerankte Evidenz E_D

1  B ← BM25(q)                         // lexikalische Kandidaten
2  V ← Vektorsuche(q)                  // semantische Kandidaten
3  F ← dedupliziere und fusioniere B, V mittels RRF
4  F ← wähle K Kandidaten unter Berücksichtigung beider Suchkanäle
5  F ← ergänze relevante benachbarte Chunks derselben Quelle
6  F' ← filtere fachlich unvereinbare Produktkandidaten, sofern genügend
        kompatible Kandidaten verbleiben
7  für jeden Chunk d in F':
8      s_CE(d) ← CrossEncoder(q, d)
9  sortiere F' nach Produktpassung, Evidenzpriorität und s_CE(d)
10 gib die ersten K_R Chunks zurück
```

Ist der Cross-Encoder nicht verfügbar, bleibt die Reihenfolge des fusionierten Retrievals erhalten. Bei einer explizit genannten PDF-Datei wird die Suche auf deren bereits indexierte Chunks begrenzt, um eine Vermischung verschiedener Dokumentversionen zu vermeiden.

## 3.8.3 Kombination von CRM- und Dokumentinformationen

Im kombinierten Pfad werden CRM- und Dokumentevidenz nicht zu einem undifferenzierten Textbestand verschmolzen. Zunächst werden ausschließlich die für die Anfrage benötigten read-only CRM-Werkzeuge ausgeführt. Die Zuordnung zwischen Kontakt, Police und Schaden wird validiert. Existieren mehrere Policen, erfolgt die Auswahl deterministisch anhand einer expliziten Policennummer, der fachlichen Produktpassung, des Status, der zeitlichen Gültigkeit und des jüngsten Vertragsbeginns.

Aus der ursprünglichen Anfrage wird anschließend eine Dokumentanfrage erzeugt. Personenbezogene Identifikatoren und reine CRM-Anweisungen werden entfernt; Produkt- und Deckungshinweise der ausgewählten Police können die Suche ergänzen. Die resultierende Evidenzmenge lautet

\[
E = E_C \cup E_D,
\]

wobei jedes Element seine Herkunftsmetadaten behält. CRM-Fakten werden im Generierungskontext ausdrücklich als strukturierte, kundenspezifische Fakten gekennzeichnet; Dokument-Chunks repräsentieren allgemeine Bedingungen. Für individuelle Vertragsmerkmale ist das CRM maßgeblich, während Deckungsregeln, Voraussetzungen und Ausschlüsse aus den Dokumenten stammen. Eine konkrete Deckung darf daher nicht allein aus einem CRM-Fakt abgeleitet werden.

**Algorithmus 3: Evidenzbildung im Combined-Pfad**

```text
Eingabe: Anfrage q und Routingplan P
Ausgabe: Antwortentwurf a sowie Evidenz E

1  X ← führe die durch P festgelegten read-only CRM-Abfragen aus
2  validiere die Entitätsbindungen in X
3  E_C ← wähle die zur Anfrage passende CRM-Evidenz
4  q_D ← entferne Identifikatoren und reine CRM-Anweisungen aus q
5  q_D ← ergänze q_D um geeignete Produkt- und Deckungshinweise aus E_C
6  E_D ← HybridRetrievalUndReranking(q_D)
7  wenn E_D wegen eines Laufzeitfehlers nicht verfügbar ist:
8      gib E_C mit Warnung zurück; leite keine Deckungsentscheidung ab
9  E ← kennzeichne und vereinige E_C und E_D unter Erhalt der Provenienz
10 a ← generiere aus q und E eine gemeinsame, quellengebundene Antwort
11 gib a und E zurück
```

## 3.8.4 Groundedness-Prüfung und Fallback-Entscheidung

Die Groundedness-Prüfung arbeitet auf Aussageebene. Aus der generierten Antwort werden semantische Claims \(c_i\) extrahiert; reine Überschriften, Quellenzeilen und Entscheidungshinweise bleiben unberücksichtigt. Für jeden Claim wird seine Unterstützung durch die bereitgestellte Evidenz bewertet. Der Support kombiniert lexikalische Überdeckung, die Übereinstimmung harter Fakten wie Policennummern, Datums-, Prozent- und Geldangaben sowie die Polarität von Deckungsaussagen.

Sei \(b\) der längengewichtete Basisscore über die jeweils am besten stützenden Chunks, \(a\) die mittlere und \(m\) die minimale Claim-Unterstützung. Der vorläufige Score der implementierten Variante `fact_aware_claim_support_v5` ist

\[
G_0=\max\left(
\min\left(b,\,0{,}72b+0{,}28m\right),
0{,}20b+0{,}55a+0{,}25m
\right).
\]

Erkannte Widersprüche begrenzen den Score zusätzlich: nicht belegte Identifikatoren auf \(0{,}20\), nicht belegte numerische Fakten auf \(0{,}38\), widersprüchliche strukturierte Felder auf \(0{,}30\), falsche Deckungspolarität auf \(0{,}25\) und unpassende Quellenangaben auf \(0{,}35\). Der endgültige Score \(G\) ist das Minimum aus \(G_0\) und allen ausgelösten Obergrenzen. Der aus der Kalibrierungsdatei geladene Akzeptanzschwellenwert beträgt im untersuchten Systemstand \(\tau=0{,}7888\).

**Algorithmus 4: Groundedness und sichere Ausgabeentscheidung**

```text
Eingabe: Anfrage q, Antwortentwurf a, Evidenz E, Schwellenwert τ
Ausgabe: finale Antwort oder Fallback

1  C ← extrahiere semantische Claims aus a
2  wenn C leer ist oder E keine nutzbare Evidenz enthält: gib FALLBACK zurück
3  berechne für jeden Claim die Evidenzunterstützung
4  berechne G_0 aus Basis-, Mittel- und Minimalsupport
5  prüfe Identifikatoren, Zahlen, strukturierte Felder, Polarität und Zitate
6  G ← begrenze G_0 durch alle ausgelösten Fehlerobergrenzen
7  wenn ein Sicherheitsverstoß oder eine ungültige Entitätsbindung vorliegt:
8      gib den zugehörigen Sicherheits-Fallback zurück
9  wenn G < τ: gib den Groundedness-Fallback zurück
10 gib a mit den zugehörigen Quellen zurück
```

Bei einem Groundedness-Fallback wird der Antwortentwurf nicht an die nutzende Person ausgegeben. Stattdessen erscheint eine neutrale Meldung, dass die verfügbare Evidenz keine hinreichend gestützte Antwort erlaubt. Schlägt im Combined-Pfad nur der Dokumententeil technisch fehl, kann eine als partiell markierte Antwort ausschließlich die validierten CRM-Fakten enthalten; eine individuelle Deckungs- oder Schadenentscheidung wird daraus nicht abgeleitet. Die Groundedness-Prüfung bewertet die Stützung vorhandener Aussagen, nicht jedoch automatisch deren fachliche Vollständigkeit. Diese beiden Qualitätsdimensionen sind daher getrennt zu evaluieren.
