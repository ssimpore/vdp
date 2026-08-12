# Model card

## Intended use

Research and engineering validation of vehicle destination prediction from
departure context, partial trajectory, and earlier completed trips.

## Not intended for

Safety-critical navigation, surveillance, enforcement, automatic denial of
service, or production claims based only on the bundled sample.

## Inputs

- Current origin and departure calendar context.
- A strictly partial observed trajectory, optionally departure-only.
- Destinations from trips completed before departure.
- Optional non-identifying vehicle context when explicitly configured.

VIN is excluded from all model tensors.

## Outputs

- Ranked destination cells with probabilities.
- Refined latitude/longitude prediction.
- Geographic error and actual candidate rank when ground truth exists.

The VIN-driven inference workspace uses a reference trip so users can start
from coherent vehicle inputs and compare with an actual destination. Users may
change departure, origin, partial trajectory, history and Top-K. The reference
actual destination remains output-side comparison data and is never supplied
to the model.

## Evaluation

Use VIN-disjoint final-test vehicles and report median, mean, and P90
Haversine error, recall by distance, and cell Top-K only when the actual cell
exists in the training vocabulary. Evaluate performance by prefix, vehicle,
history availability, destination frequency, geography, and time period.

## Privacy

Trips and destinations are sensitive mobility data. Apply access control,
retention limits, encryption, pseudonymization, aggregation, and applicable
legal review. The UI displays pseudonymized vehicle identifiers in evaluation
tables, while source VINs remain available only for local grouping and splits.
