# RAG Fix Validation

## Runtime Results

```json
{
  "planned_query_count": 100,
  "complete": false,
  "query_count": 25,
  "before_http_status": {
    "503": 100
  },
  "after_http_status": {
    "200": 19,
    "504": 6
  },
  "after_runtime_outcomes": {
    "runtime_error": 6,
    "answered": 7,
    "fallback": 12
  },
  "evaluator_failures": {
    "unfaithful_claim_extraction": 9,
    "incomplete_claim_extraction": 1
  },
  "hallucination_rate": null,
  "limitations": "No independent factual adjudication: answered is not a correctness label. Evaluator failures and fallback are not confirmed hallucinations. Baseline and rerun use different concurrency; latency is not directly comparable."
}
```

## All Questions and Answers

### pdf-rag-001

Question: What coverage or rule is described for "Grounding, strike or bankruptcy of your airline or tour operator" in assistance brochure?

Reference: Grounding, strike or bankruptcy of your airline or tour operator Insured benefits We will cover the cost if you have to cancel, abort or extend your trip.

Delivered answer: 

Runtime outcome: runtime_error; evaluator: None; failure: None

Reference evidence:
- assistance-brochure.pdf, page 4: Grounding, strike or bankruptcy of your airline or tour operator Insured benefits We will cover the cost if you have to cancel, abort or extend your trip.

### pdf-rag-002

Question: How does assistance brochure describe "Cancellation cost insurance Cancellation cost insurance covers you for unforeseen"?

Reference: Cancellation cost insurance Cancellation cost insurance covers you for unforeseen events occurring before your departure that make it impossible for you to set off on your journey.

Delivered answer: Cancellation cost insurance covers you for unforeseen events occurring before your departure that make it impossible for you to set off on your journey. 

Source: [assistance-brochure.pdf, page 2]

Runtime outcome: answered; evaluator: success; failure: None

Reference evidence:
- assistance-brochure.pdf, page 2: Cancellation cost insurance Cancellation cost insurance covers you for unforeseen events occurring before your departure that make it impossible for you to set off on your journey.

### pdf-rag-003

Question: What does assistance brochure say concerning "“All-round insurance: in your leisure time, while travelling, worldwide.” Good"?

Reference: “All-round insurance: in your leisure time, while travelling, worldwide.” Good to know With Assistance for persons insurance, all booked holidays, trips, events and concerts are insured throughout the year.

Delivered answer: The assistance brochure states that with Assistance for persons insurance, all booked holidays, trips, events, and concerts are insured throughout the year. This coverage applies to leisure time and while travelling worldwide. 

Source:
- [assistance-brochure.pdf, page 3]

Runtime outcome: answered; evaluator: success; failure: None

Reference evidence:
- assistance-brochure.pdf, page 3: “All-round insurance: in your leisure time, while travelling, worldwide.” Good to know With Assistance for persons insurance, all booked holidays, trips, events and concerts are insured throughout the year.

### pdf-rag-004

Question: According to assistance brochure, what does it state about "Cancellation cost insurance covers you for unforeseen events occurring before"?

Reference: Cancellation cost insurance covers you for unforeseen events occurring before your departure that make it impossible for you to set off on your journey.

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-brochure.pdf, page 4: Cancellation cost insurance covers you for unforeseen events occurring before your departure that make it impossible for you to set off on your journey.

### pdf-rag-005

Question: What coverage or rule is described for "Under the motor vehicle Assistance policy, cover is provided in" in assistance brochure?

Reference: Under the motor vehicle Assistance policy, cover is provided in Switzerland, the Principality of Liechtenstein and in European countries under the “International Motor Insurance Card” agreement (Green Card).

Delivered answer: 

Runtime outcome: runtime_error; evaluator: None; failure: None

Reference evidence:
- assistance-brochure.pdf, page 6: Under the motor vehicle Assistance policy, cover is provided in Switzerland, the Principality of Liechtenstein and in European countries under the “International Motor Insurance Card” agreement (Green Card).

### pdf-rag-006

Question: How does assistance brochure describe "Assistance for persons Assistance for persons provides insurance to cover"?

Reference: Assistance for persons Assistance for persons provides insurance to cover you if anything happens while you're travelling.

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: uncertain; failure: None

Reference evidence:
- assistance-brochure.pdf, page 2: Assistance for persons Assistance for persons provides insurance to cover you if anything happens while you're travelling.

### pdf-rag-007

Question: What does assistance brochure say concerning "Assistance for persons provides insurance to cover you if anything"?

Reference: Assistance for persons provides insurance to cover you if anything happens while you're travelling.

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-brochure.pdf, page 4: Assistance for persons provides insurance to cover you if anything happens while you're travelling.

### pdf-rag-008

Question: According to assistance brochure, what does it state about "If you are unable to continue your journey due to"?

Reference: If you are unable to continue your journey due to illness or an accident Insured benefits

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-brochure.pdf, page 5: If you are unable to continue your journey due to illness or an accident Insured benefits

### pdf-rag-009

Question: What coverage or rule is described for "Cover for e-bikes and motorcycles is valid in Switzerland and" in assistance brochure?

Reference: Cover for e-bikes and motorcycles is valid in Switzerland and the Principality of Liechtenstein as well as in neighbouring countries within a distance of 150 km (as the crow flies) of the common border.

Delivered answer: The coverage for e-bikes and motorcycles is valid in:

- Switzerland
- Neighbouring countries within a distance of 150 km (as the crow flies) of the common border.

This is specified under the section discussing territorial validity for e-bikes and motorcycles.

Source: [assistance-brochure.pdf, page 6]

Runtime outcome: answered; evaluator: success; failure: None

Reference evidence:
- assistance-brochure.pdf, page 6: Cover for e-bikes and motorcycles is valid in Switzerland and the Principality of Liechtenstein as well as in neighbouring countries within a distance of 150 km (as the crow flies) of the common border.

### pdf-rag-010

Question: How does assistance brochure describe "Motor vehicle Assistance If you travel by car or motorbike"?

Reference: Motor vehicle Assistance If you travel by car or motorbike, or occasionally rent a vehicle, it makes sense to take out motor vehicle Assistance insurance.

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-brochure.pdf, page 7: Motor vehicle Assistance If you travel by car or motorbike, or occasionally rent a vehicle, it makes sense to take out motor vehicle Assistance insurance.

### pdf-rag-011

Question: What does assistance sti say concerning "C5 Bias of the assessor The insurance does not cover"?

Reference: C5 Bias of the assessor The insurance does not cover events in which the assessor (expert, doctor, assessor, etc.) who is evaluating the loss occurrence directly benefits from the loss occurrence or is related to the insured person by blood or marriage.

Delivered answer: The insurance does not cover events in which the assessor (expert, doctor, assessor, etc.) who is evaluating the loss occurrence directly benefits from the loss occurrence or is related to the insured person by blood or marriage. 

Source: [assistance-sti.pdf, page 3]

Runtime outcome: answered; evaluator: success; failure: None

Reference evidence:
- assistance-sti.pdf, page 3: C5 Bias of the assessor The insurance does not cover events in which the assessor (expert, doctor, assessor, etc.) who is evaluating the loss occurrence directly benefits from the loss occurrence or is related to the insured person by blood or marriage.

### pdf-rag-012

Question: According to assistance sti, what does it state about "B2 Delayed commencement of travel If, as a result of"?

Reference: B2 Delayed commencement of travel If, as a result of an insured event, the insured person cannot take a trip or enter a rented property or event until after the originally agreed date, then instead of covering the cancellation costs, Helvetia will cover the additional costs incurred in relation to the delayed departure and the costs for the unused part of the stay in proportion to the arrangement price (not including transport costs).

Delivered answer: 

Runtime outcome: runtime_error; evaluator: None; failure: None

Reference evidence:
- assistance-sti.pdf, page 3: B2 Delayed commencement of travel If, as a result of an insured event, the insured person cannot take a trip or enter a rented property or event until after the originally agreed date, then instead of covering the cancellation costs, Helvetia will cover the additional costs incurred in relation to the delayed departure and the costs for the unused part of the stay in proportion to the arrangement price (not including transport costs).

### pdf-rag-013

Question: What coverage or rule is described for "C6 Non-specialist medical assessment The insurance does not cover events" in assistance sti?

Reference: C6 Non-specialist medical assessment The insurance does not cover events in which medical certificates, expert opinions or comparable medical assessments are issued by a specialist whose speciality is not factually related to the health related loss occurrence claimed.

Delivered answer: 

Runtime outcome: runtime_error; evaluator: None; failure: None

Reference evidence:
- assistance-sti.pdf, page 3: C6 Non-specialist medical assessment The insurance does not cover events in which medical certificates, expert opinions or comparable medical assessments are issued by a specialist whose speciality is not factually related to the health related loss occurrence claimed.

### pdf-rag-014

Question: How does assistance sti describe "However, we will grant the insured persons coverage if these"?

Reference: However, we will grant the insured persons coverage if these faults could not have been recognised even with due attention.

Delivered answer: 

Runtime outcome: runtime_error; evaluator: None; failure: None

Reference evidence:
- assistance-sti.pdf, page 7: However, we will grant the insured persons coverage if these faults could not have been recognised even with due attention.

### pdf-rag-015

Question: What does assistance sti say concerning "A3 Failure and delays If, either before beginning a journey"?

Reference: A3 Failure and delays If, either before beginning a journey or while en route to the point of departure of a booked trip, the insured person experiences one of the following events: a) failure or delay of public transport or means of transportation; b) the vehicle or taxi used for transportation cannot be driven (is unroadworthy) due to breakdown, accident, theft, fire or natural forces.

Delivered answer: 

Runtime outcome: runtime_error; evaluator: None; failure: None

Reference evidence:
- assistance-sti.pdf, page 3: A3 Failure and delays If, either before beginning a journey or while en route to the point of departure of a booked trip, the insured person experiences one of the following events: a) failure or delay of public transport or means of transportation; b) the vehicle or taxi used for transportation cannot be driven (is unroadworthy) due to breakdown, accident, theft, fire or natural forces.

### pdf-rag-016

Question: According to assistance sti, what does it state about "H4 Search, rescue and recovery costs Search, rescue and recovery"?

Reference: H4 Search, rescue and recovery costs Search, rescue and recovery costs are paid for the insured person.

Delivered answer: The insurance covers search, rescue, and recovery costs for the insured person. 

Source: [assistance-sti.pdf, page 7]

Runtime outcome: answered; evaluator: success; failure: None

Reference evidence:
- assistance-sti.pdf, page 7: H4 Search, rescue and recovery costs Search, rescue and recovery costs are paid for the insured person.

### pdf-rag-017

Question: What coverage or rule is described for "In addition, under age children travelling in the care of" in assistance sti?

Reference: In addition, under age children travelling in the care of an insured person are also covered.

Delivered answer: The coverage described states that under-age children traveling in the care of an insured person are also covered. This applies depending on what is agreed in the policy, which may cover either the policyholder (one-person household) or the policyholder and all persons cohabiting with the policyholder (multi-person household) [assistance-sti.pdf, page 9].

Runtime outcome: answered; evaluator: success; failure: None

Reference evidence:
- assistance-sti.pdf, page 9: In addition, under age children travelling in the care of an insured person are also covered.

### pdf-rag-018

Question: How does assistance sti describe "E1 Call-back and transport costs D5 Damage to property at"?

Reference: E1 Call-back and transport costs D5 Damage to property at the residential address If property of the insured person at their permanent residence or a second residence is significantly affected by a fire, natural forces, water event or theft event and the insured person is unable to continue the trip or holiday as planned.

Delivered answer: I cannot include or expose personal or sensitive information in the response.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-sti.pdf, page 5: E1 Call-back and transport costs D5 Damage to property at the residential address If property of the insured person at their permanent residence or a second residence is significantly affected by a fire, natural forces, water event or theft event and the insured person is unable to continue the trip or holiday as planned.

### pdf-rag-019

Question: What does assistance sti say concerning "CHF 2ʼ000 per insured person if the insured person requires"?

Reference: CHF 2ʼ000 per insured person if the insured person requires an unfore seen stay, an unfo reseen extension to their stay, or must acquire more suita ble accommodation for care purposes; Max.

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: incomplete_claim_extraction

Reference evidence:
- assistance-sti.pdf, page 4: CHF 2ʼ000 per insured person if the insured person requires an unfore seen stay, an unfo reseen extension to their stay, or must acquire more suita ble accommodation for care purposes; Max.

### pdf-rag-020

Question: According to assistance sti, what does it state about "Spare parts are deemed to be only those parts ordinarily"?

Reference: Spare parts are deemed to be only those parts ordinarily transported by breakdown assistance vehicles (fuel and vehicle batteries are not insured).

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-sti.pdf, page 7: Spare parts are deemed to be only those parts ordinarily transported by breakdown assistance vehicles (fuel and vehicle batteries are not insured).

### pdf-rag-021

Question: What coverage or rule is described for "K6 Transport and additional transport costs The costs incurred by" in assistance sti?

Reference: K6 Transport and additional transport costs The costs incurred by the insured person to return home or to continue the journey by public transport or taxi to his/her place of work, place of residence or temporary holiday domicile.

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-sti.pdf, page 8: K6 Transport and additional transport costs The costs incurred by the insured person to return home or to continue the journey by public transport or taxi to his/her place of work, place of residence or temporary holiday domicile.

### pdf-rag-022

Question: How does assistance sti describe "The benefits are limited to the amount of the agreed"?

Reference: The benefits are limited to the amount of the agreed arrangement price.

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: uncertain; failure: None

Reference evidence:
- assistance-sti.pdf, page 3: The benefits are limited to the amount of the agreed arrangement price.

### pdf-rag-023

Question: What does assistance sti say concerning "These benefits cannot be combined with the benefits from personal"?

Reference: These benefits cannot be combined with the benefits from personal assistance or cancellation costs.

Delivered answer: The statement indicates that the benefits covered by the insurance cannot be combined with the benefits from personal assistance or cancellation costs. This means that if a claim is made under one category, it cannot also receive benefits from the other categories for the same event. 

Source: [assistance-sti.pdf, page 7]

Runtime outcome: answered; evaluator: success; failure: None

Reference evidence:
- assistance-sti.pdf, page 7: These benefits cannot be combined with the benefits from personal assistance or cancellation costs.

### pdf-rag-024

Question: According to assistance sti, what does it state about "the breakage of vehicle parts made of glass or materials"?

Reference: the breakage of vehicle parts made of glass or materials used to replace glass as well as light bulbs, to the extent that they are destroyed by the glass breakage; ■ tyre damage; ■ damage to or loss of car keys, including replacement of the lock cylinder where necessary.

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-sti.pdf, page 7: the breakage of vehicle parts made of glass or materials used to replace glass as well as light bulbs, to the extent that they are destroyed by the glass breakage; ■ tyre damage; ■ damage to or loss of car keys, including replacement of the lock cylinder where necessary.

### pdf-rag-025

Question: What coverage or rule is described for "For the costs for the unused part of the stay" in assistance sti?

Reference: For the costs for the unused part of the stay in proporti on to the arran gement price (not including transport costs).

Delivered answer: I cannot provide a safe, policy-compliant answer for this request. Please rephrase.

Runtime outcome: fallback; evaluator: failed; failure: unfaithful_claim_extraction

Reference evidence:
- assistance-sti.pdf, page 4: For the costs for the unused part of the stay in proporti on to the arran gement price (not including transport costs).
