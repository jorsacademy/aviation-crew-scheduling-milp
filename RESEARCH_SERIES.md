# Aviation Operations Research Series

This file maps aviation-related optimization repositories. It is an index only: each repository remains independent because crew, aircraft, disruption, maintenance, and airport-service decisions have different state spaces, constraints, and solution methods.

## Crew planning and scheduling

- `aviation-crew-scheduling-milp` — qualification-, duty-, rest-, and mission-aware MILP.
- `airline-crew-scheduling-column-generation` — crew scheduling through column generation and pricing.
- `airline-crew-workforce-optimization-ga` — crew/workforce optimization using a genetic algorithm.

These three should remain separate: they expose different computational paradigms for related personnel-planning problems.

## Aircraft and fleet decisions

- `aircraft-allocation-under-uncertain-demand` — aircraft/fleet allocation under demand uncertainty.
- `intercity-rail-fleet-circulation-optimization` — not aviation, but a useful cross-domain fleet-circulation analogue.
- `fleet-decarbonization-optimizer` — long-horizon fleet-transition decisions; cross-domain rather than aviation-specific.

## Maintenance and recovery

- `aircraft-maintenance-scheduling-gurobi` — maintenance scheduling.
- `airline-disruption-recovery-gurobi` — post-disruption recovery decisions.
- `airline-operations-under-uncertainty-stochastic-optimization` — ex-ante stochastic planning under uncertain operations.

The maintenance, recovery, and stochastic-planning repositories are complementary rather than duplicate: one schedules planned maintenance, one reacts to realized disruption, and one plans before uncertain outcomes are known.

## Airport service and capacity

- `airport-checkin-counter-optimization-erlang-c` — analytical queueing/staffing perspective.
- `airport-checkin-simulation-optimization` — simulation-based airport check-in optimization.
- `metroglobal-airport-counter-optimization` — airport-counter optimization with a separate application setup.

## Portfolio rule

Aviation is the application domain, not the consolidation key. Repositories should remain separate when the decision layer changes materially — for example crew pairing versus aircraft assignment, planned maintenance versus disruption recovery, or queueing formulas versus simulation optimization.
