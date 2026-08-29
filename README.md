# Aviation Crew Scheduling MILP

A synthetic mixed-integer linear programming (MILP) example for aviation crew scheduling using Python and PuLP.

The model assigns pilots, co-pilots, and flight engineers to missions while accounting for crew qualifications, availability, duty-hour limits, rest rules, mission priority, combat/night qualifications, overtime, hazard pay, and mission-completion penalties.

## Scope

This repository is intended for operations-research education and demonstration. The data and rules are synthetic. It is not an operational aviation, airline, military, or regulatory crew-planning system.

## Model overview

The optimizer includes:

- binary mission-completion decisions,
- crew-to-aircraft assignment variables,
- pilot/co-pilot/engineer composition constraints,
- combat and night qualification constraints,
- personnel availability constraints,
- one-mission-per-day assignment logic,
- daily and weekly duty-hour limits,
- regular-time and overtime accounting,
- next-day rest constraints,
- stricter rest after combat-qualified work,
- mandatory completion of critical missions,
- minimum completion requirements for high-priority missions,
- labor, hazard-pay, and uncompleted-mission penalty costs.

The objective minimizes total labor cost, overtime cost, hazard pay, and penalties for missions that are not completed.

## Project structure

```text
aviation-crew-scheduling-milp/
├── README.md
├── requirements.txt
└── crew_scheduling.py
```

## Requirements

- Python 3.10+
- PuLP
- NumPy
- openpyxl (for Excel export)

Install dependencies:

```bash
pip install -r requirements.txt
```

PuLP's bundled CBC solver is the expected default. If Gurobi or CPLEX command-line solvers are installed and available, the code will try them before CBC.

## Run

```bash
python crew_scheduling.py
```

The script reports the number of variables and constraints, solves the model, prints mission assignments and cost components, then runs workload, completion-rate, and scenario analyses.

## Scenario analysis

The included sensitivity/scenario analysis re-solves modified models rather than applying artificial percentage changes to the base objective. Current scenarios include:

- removing two pilots,
- doubling mission non-completion penalties,
- relaxing rest constraints.

## Export results

The module exposes helper functions for CSV and Excel export:

```python
from crew_scheduling import (
    CrewSchedulingOptimizer,
    export_results_to_csv,
    export_results_to_excel,
)

optimizer = CrewSchedulingOptimizer()
optimizer.solve()

export_results_to_csv(optimizer)
export_results_to_excel(optimizer)
```

## Important modeling assumption

Mission start and end times are not included in the synthetic input data. Therefore, the model conservatively assumes that a crew member may serve at most one mission per day.

## Educational limitations

A real crew-scheduling system would typically require substantially more detail, including exact duty windows, legal rest periods, qualification expiry, recurrent training, aircraft type ratings, reserve crews, positioning/deadheading, base locations, pairing construction, disruptions, labor agreements, and jurisdiction-specific aviation regulations.
