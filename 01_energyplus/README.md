# EnergyPlus model and weather

This directory contains the EnergyPlus configuration associated with the completed July single-setpoint experiment and the Omaha TMY3 weather file.

| File | Purpose |
| --- | --- |
| `HIBO_LabB_July_Single_Setpoint_Excitation.idf` | Building geometry, constructions, HVAC, schedules, and requested outputs |
| `USA_NE_Omaha-Eppley.Airfield.725500_TMY3.epw` | Omaha Eppley Airfield, station 725500; local standard time UTC−6 |
| `provenance.json` | Input hashes, dependency inspection, and weather consistency checks |

## Simulation configuration

- EnergyPlus **22.2.0**; the archived output records build `c249759bad`.
- Weather run: July 1–31, with July 1 defined as Sunday and daylight-saving adjustment disabled.
- Zone timestep: 60 timesteps per hour, equivalent to one minute.
- Target zone: `Thermal Zone: Space 103 LAB B`.
- Lab B uses `ThermostatSetpoint:SingleHeatingOrCooling` during the July weather run. The common heating/cooling target takes the values 20, 22, 24, 26, and 28°C.
- Other conditioned zones retain their existing thermostat schedules. Shared AHU operation and interzone heat transfer remain active.

All 16 schedules are embedded `Schedule:Compact` objects. Inspection of the active IDF objects found no `Schedule:File`, Python plugin, external interface, external table, or HVAC template dependencies. No additional control script or schedule file is required by this configuration.

## Run EnergyPlus

From the repository root, with the EnergyPlus 22.2 executable on `PATH`:

```bash
energyplus -r -w "01_energyplus/USA_NE_Omaha-Eppley.Airfield.725500_TMY3.epw" -d "runs/energyplus_july" "01_energyplus/HIBO_LabB_July_Single_Setpoint_Excitation.idf"
```

In Windows PowerShell, an explicit installation path can be used:

```powershell
& "C:\EnergyPlusV22-2-0\energyplus.exe" -r -w "01_energyplus\USA_NE_Omaha-Eppley.Airfield.725500_TMY3.epw" -d "runs\energyplus_july" "01_energyplus\HIBO_LabB_July_Single_Setpoint_Excitation.idf"
```

The `-r` option runs ReadVarsESO to produce CSV output. Review the generated `eplusout.err` and the output timestamps after a new simulation. This repository release does not rerun EnergyPlus or replace the archived calibration observations.

## Connection to the thermal model

The IDF supplies surface geometry, construction layers, and material properties for the initial 17R8C reference model. The prior mean uses the resulting model reference values; the prior standard deviation is a modeling choice equal to twice that mean. Resistance splitting and surface aggregation are additional RC-model assumptions, not independent values reported directly by EnergyPlus.

The authoritative input used by the completed inference is retained at:

- `../03_workflow/initial/inputs/HIBO.idf`
- `../03_workflow/initial/inputs/HIBO_MCMC_1min.csv`
- `../03_workflow/initial/inputs/reference.json`
- `../03_workflow/initial/inputs/data.npz`

The CSV covers all 44,640 July minutes. The thermal workflow uses July 1 for conditioning, July 2–22 for training, and July 23–31 for validation and model selection. Its physical input interval is one minute; likelihood evaluation occurs every five minutes.

## Provenance and verification limits

The published IDF contains the same 513 active objects and field values as the IDF archived in the completed inference package. Three historical instructions were removed from comments, and six material-source comments were rewritten in neutral wording. Technical source URLs, material assumptions, and all simulation inputs are preserved. Both published IDF copies are byte-identical to each other.

| IDF record | SHA-256 |
| --- | --- |
| Archived inference input | `80bdbd484400f78263fef5fbd0cde3b5f2a6bf9f589cbbf229215580a47754f4` |
| Published comment-cleaned input | `c2bd736fcba626f9ff652ba1fb4548f047b66a567eaf121e9dc0aa0b4bb9cede` |

The comment-only transformation and canonical active-object comparison are recorded in `provenance.json`. The original EnergyPlus output did not record a hash of the executed IDF, so exact executed-file identity cannot be established independently from those outputs alone.

The EPW was recovered from the project weather files. Every July hourly outdoor dry-bulb value matches the archived inference input exactly. Linear interpolation reproduces all 44,640 one-minute outdoor temperatures to a maximum difference of approximately `1.42e-14`°C when the initial previous-hour value is taken from July 1 at hour 24, consistent with the same-day warmup boundary. This verifies consistency of the weather temperatures; an original executed-EPW hash was not recorded.

The archived simulation audit verifies full-month coverage and agreement between CSV and SQL thermostat/temperature outputs. It also records that the supplied SQL completion flags are false and that no `.err` file accompanied that run. Those limitations remain part of the input record.
