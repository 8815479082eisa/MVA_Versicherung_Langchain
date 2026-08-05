# Human-Review-Protokoll für Groundedness

## Rollen

Zwei Reviewer bearbeiten ihre CSV unabhängig. Die Blind-Map bleibt bis zum Abschluss beider Reviews verborgen.

## Label

- `SUPPORTED`: Jede sachliche Behauptung der Candidate Answer ist durch den Context gestützt.
- `UNSUPPORTED`: Mindestens eine sachliche Behauptung widerspricht dem Context oder ist darin nicht belegt.
- `AMBIGUOUS`: Der Context reicht für eine verlässliche Entscheidung nicht aus.

## Fehlerkategorie

Falls `UNSUPPORTED`: `wrong_customer`, `wrong_policy`, `wrong_current_policy`, `wrong_status`, `wrong_date`, `wrong_number`, `wrong_premium`, `wrong_deductible`, `wrong_coverage`, `wrong_limit`, `wrong_percentage`, `wrong_citation`, `negation_or_exclusion`, `unsupported_claim`, `other`.

## Criticality

`HIGH`, wenn eine falsche Kunden-/Policenzuordnung, Deckungsentscheidung, Prämie, Selbstbeteiligung, Leistungslimite oder Ausschlussaussage unmittelbare Versicherungswirkung haben könnte; sonst `MEDIUM` oder `LOW`. Dies ist erst nach menschlicher Bewertung eine Human-Annotation.

## Confidence

`HIGH`, `MEDIUM` oder `LOW`. Jeder Reviewer dokumentiert Unsicherheit in `notes`.

## Adjudikation

Nach beiden Reviews werden Übereinstimmung und Konflikte berechnet. Konflikte und alle `AMBIGUOUS`-Fälle werden gemeinsam adjudiziert. Weak Labels, v4/v5-Scores und Mutationsmetadaten dürfen erst danach eingeblendet werden.
