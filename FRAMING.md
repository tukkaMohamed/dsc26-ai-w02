# Framing: Rawaj Pay mule detection

## Decision

The model's score helps Rawaj Pay decide which transfers should be reviewed as potential mule activity. Dina, Head of Fraud, needs to prioritize cases because the team has only five analysts and limited review capacity. False positives waste analysts' time, while false negatives allow suspicious transfers to go undetected.

## Metric

Precision and recall should be considered together, alongside the operational cost of false positives and false negatives. The threshold should minimize expected cost while keeping the number of flagged transfers manageable for the analysts. Accuracy alone is not sufficient because mule cases may be much less common than legitimate transfers.

## Baseline

The baseline is Rawaj Pay's existing process before using a model, such as its current rules and manual review. The model should be compared against this baseline using the same data and business costs. It should improve the detection of mule activity without overwhelming the five analysts with unnecessary alerts.

## Classical ML or LLM

A classical machine learning model is the better starting point because mule detection relies on structured transfer data and measurable outcomes. A rules-based baseline can provide a simple comparison. An LLM is not necessary for scoring structured transactions, although it could potentially help analysts summarize case information later.