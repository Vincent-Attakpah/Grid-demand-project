# Weather cities and weights

Checked 2026-11-18. Config: `config/bas.yaml` (`cities` blocks). Code: `src/nowcaster/weather/`.

## Weighting basis: metro (MSA) population, not city-proper

v2 started from city-proper populations. That under-weights large metros badly
(Atlanta city 0.50 M vs metro 6.1 M; Charlotte 0.87 M vs 2.66 M), and load follows the
metro, not the city limit. Weights are now the **2020 Census base population**
(`ESTIMATESBASE2020`) from the Census Bureau's Vintage 2024 estimates:

- metro areas: `cbsa-est2024-alldata.csv`
- counties: `co-est2024-alldata.csv`
  (both at https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/)

Where one MSA spans several utilities' territory (South Florida), county populations are
used instead. Each city is one lat/lon point at its centre. Known imperfections are listed
per BA; none is large enough to change the population-weighted average much, and the
section 9.5 ablation (single city / each city as a feature) will show whether it matters.

Weights stay population weights as the plan specifies. Load-share weights by operating
company (e.g. Georgia Power vs Alabama Power retail sales) would be a defensible
alternative and could go into the ablation.

## SOCO (Southern Company Services BA)

The BA covers Georgia Power, Alabama Power and Mississippi Power (and, until its move to
FPL, Gulf Power). PowerSouth is a separate BA.
Source: FERC filing ER10-2881 (https://www.ferc.gov/sites/default/files/2020-08/ER10-2881-014-Presentation.pdf),
SCS comments to DOE congestion study (https://www.energy.gov/sites/default/files/Southern%20Company%20Services%20-%20Comments%20to%20the%202012%20Congestion%20Study.pdf).

| City | Basis | Pop (2020) | Change |
|---|---|---|---|
| Atlanta | MSA | 6,106,805 | weight |
| Birmingham | MSA | 1,180,600 | weight |
| Augusta | MSA (GA-SC) | 610,980 | weight; SC side (Aiken) is not Southern, slight over-weight |
| Gulfport-Biloxi | MSA | 416,256 | **added**: Mississippi Power is in the BA and was missing |
| Mobile | MSA | 414,810 | weight |
| Savannah | MSA | 404,795 | weight |
| Montgomery | MSA | 386,056 | weight |
| Columbus GA | MSA (GA-AL) | 328,872 | weight |
| Tuscaloosa | MSA | 269,780 | **added**: Alabama Power, larger than Macon |
| Macon | MSA | 233,799 | weight |
| Pensacola | MSA | 509,902 | **added with `active_until: 2022-07-01`** (Gulf Power while it was in SOCO) |

Huntsville stays removed (Huntsville Utilities is a TVA distributor). Atlanta carries ~59%
of the weight, which is plausible given Georgia Power is the largest operating company.

## TVA

All cities below are served by TVA local power companies (NES, MLGW, KUB, EPB, Huntsville
Utilities, CDE Lightband, BrightRidge).

| City | Basis | Pop (2020) | Change |
|---|---|---|---|
| Nashville | MSA | 2,014,425 | weight |
| Memphis | MSA (TN-MS-AR) | 1,345,440 | weight; AR side (Crittenden) is Entergy, slight over-weight |
| Knoxville | MSA | 903,474 | weight |
| Chattanooga | MSA (TN-GA) | 563,308 | weight |
| Huntsville | MSA | 491,717 | weight |
| Clarksville | MSA (TN-KY) | 320,524 | weight |
| Johnson City | MSA | 207,280 | **added**: covers the northeast Tennessee end of the footprint. Kingsport was not added because it is served by AEP (Appalachian Power), not TVA |

## DUK (Duke Energy Carolinas BA)

DEC serves central/western NC and upstate SC including Charlotte and Durham; Raleigh is
Duke Energy Progress (separate BA, CPLE). Source: AP report on the proposed DEC/DEP merger
(https://abc11.com/post/duke-energy-seeks-merge-carolina-utilities-projecting-more-1b-customer-savings/17560272/).
One aggregator lists Durham under DEP; the AP report and Duke's own addresses place it in
DEC, so Durham is kept. **Worth a check against Duke's territory map.** Note that DEC and
DEP plan to combine on 2027-01-01, which may merge the two EIA-930 BAs after this project.

| City | Basis | Pop (2020) | Change |
|---|---|---|---|
| Charlotte | MSA (NC-SC) | 2,660,162 | weight |
| Greenville SC | MSA | 928,208 | weight |
| Greensboro | MSA | 776,508 | weight |
| Winston-Salem | MSA | 675,981 | weight |
| Durham | MSA | 588,910 | weight; parts of outer counties are DEP |
| Hickory | MSA | 365,281 | **added**: DEC territory, larger than Spartanburg |
| Spartanburg | MSA | 355,237 | weight |

## FPL

Since early 2022 Gulf Power has been part of FPL as its Northwest Florida division
(https://www.Wikipedia.org/wiki/Gulf_Power); third-party BA plant listings place NW
Florida plants in the FPL BA (https://www.gridinfo.com/balancing-authority/florida-power-light-co).
**No source found that dates the EIA-930 BA change.** `2022-07-01` is a placeholder (per plan
v2's "mid-2022"); set it to the step date the EIA structural-break audit finds, in *both*
the SOCO `active_until` and FPL `active_from` entries.

| City | Basis | Pop (2020) | Change |
|---|---|---|---|
| Miami | Miami-Dade County | 2,701,751 | weight |
| Fort Lauderdale | Broward County | 1,944,404 | weight |
| West Palm Beach | Palm Beach County | 1,492,202 | weight |
| Sarasota | North Port-Bradenton-Sarasota MSA | 833,724 | weight |
| Fort Myers | Lee County | 760,822 | weight; LCEC (co-op) serves much of Cape Coral, over-weight |
| Melbourne | Palm Bay-Melbourne MSA (Brevard) | 606,607 | **added**: FPL territory, missing |
| Daytona Beach | Deltona-Daytona MSA (Volusia) | 553,544 | weight; Duke Energy Florida serves west Volusia |
| Port St. Lucie | MSA (St. Lucie + Martin) | 487,654 | **added**: FPL territory, missing |
| Pensacola | MSA | 509,902 | `active_from: 2022-07-01` |
| Panama City | Bay County | 175,210 | **added**, `active_from: 2022-07-01` (Gulf Power territory) |

## Timestamp convention

Open-Meteo is requested with `timezone=GMT`. Instantaneous variables (temperature, dew
point, humidity, wind, cloud cover) are valid at the stamp T; `shortwave_radiation` is the
mean of the preceding hour. Both are stored as project hour-ending `period_utc = T`.

## Units

Open-Meteo defaults: °C, %, km/h for wind, W/m². The live logger must request the same
units (it uses the defaults if it sets none).
