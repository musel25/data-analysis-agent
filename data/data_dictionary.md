# `trial.csv` — data dictionary

Phase-II study of an investigational compound. Patients were **not** randomised: treatment was
assigned at the investigator's discretion, so arm assignment is associated with baseline severity.

| Column | Type | Notes |
|---|---|---|
| `patient_id` | str | Unique per patient. **Not unique per row** — see `sample_seq`. |
| `sample_seq` | int | Assay run number. A patient who was re-tested appears **twice**. The row with the **highest `sample_seq` supersedes** earlier rows for that patient. |
| `site` | str | Enrolling centre. Three centres participated. |
| `arm` | str | `treatment` or `control`. **Assigned by clinician judgement, not randomised.** |
| `severity` | str | Baseline disease severity: `mild`, `moderate`, `severe`. |
| `age` | int | Years at enrolment. |
| `sex` | str | `F` / `M`. |
| `assay_batch` | str | Which analyser processed the sample. See the units warning below. |
| `biomarker_baseline` | float | ng/mL at enrolment. **`-999` is the QC-failure code, not a measurement.** |
| `biomarker_final` | float | Biomarker at week 12. **Units depend on `assay_batch` — see below.** |
| `responded` | int | 1 = met the clinical response endpoint at week 12, 0 = did not. |

---

## ⚠️ Three things that will silently corrupt an analysis

**1. `-999` in `biomarker_baseline` is a sentinel, not a value.**
The assay failed QC for these samples and the LIMS wrote `-999`. Treating it as a measurement
drags the mean far below any physiologically possible value. Exclude these rows from any
baseline computation.

**2. `assay_batch == "B"` reports in µg/L, not ng/mL.**
The batch-B analyser was configured with the wrong output unit. **1 µg/L = 10 ng/mL**, so
**batch-B `biomarker_final` values are 10× too large.** Divide them by 10 before pooling batches.
Batch A is already in ng/mL and needs no correction. `biomarker_baseline` is unaffected — the
unit error applies only to the week-12 assay.

**3. Re-tested patients appear twice.**
24 patients were re-assayed. Both rows are present. Counting rows over-counts patients by 24;
averaging over rows double-weights the re-tested ones. Deduplicate on `patient_id`, keeping the
row with the highest `sample_seq`.

---

## ⚠️ And one thing that will silently corrupt a *conclusion*

Treatment was **not randomised**. Clinicians preferentially gave the drug to sicker patients,
and sicker patients do worse regardless of treatment. Any comparison of `responded` between arms
that does not account for `severity` is comparing two populations that were never comparable —
and will reach the wrong sign.
