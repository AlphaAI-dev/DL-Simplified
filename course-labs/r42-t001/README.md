# R42-T001 · Fashion-MNIST – CNN · Dataset Contract (Schema Validator)

**Student:** Rohit Kumar Rai  
**Task:** R42-T001 · 1 of 10  
**Branch:** `R42-T001`  
**Closes:** [S-AI-Projects/DL-Simplified#1](https://github.com/S-AI-Projects/DL-Simplified/issues/1)

---

## ⚠️ Data Notice

> This component uses a **synthetic CPU smoke fixture** (six 4×4 grayscale images).  
> It is **not** the Fashion-MNIST dataset and must not be used to make performance claims about Fashion-MNIST or any real-world task.

---

## Goal

Implement a schema validator for image input records with fields:

| Field | Type | Constraint |
|-------|------|-----------|
| `record_id` | `str` | non-empty, unique |
| `group_id` | `str` | non-empty |
| `pixels` | `list[int]` | flat list, length == 16 (4×4, row-major), each value an `int` in \[0, 255\] (`bool`/`float` rejected) |
| `label` | `int` | in `{0, 1}` (`bool` rejected) |

Failures are **always reported** as `ValidationFailure(record_id, field, reason)` — records are never silently dropped.
For example, a duplicate ID yields `record_id='r001', field='record_id', reason="duplicate record_id 'r001'"`.

---

## Synthetic Fixture

| record_id | group_id | pixels | label |
|-----------|----------|--------|-------|
| r001 | g1 | all-zero (4×4) | 0 |
| r002 | g1 | all-zero (4×4) | 0 |
| r003 | g2 | all-zero (4×4) | 0 |
| r004 | g2 | white diagonal (4×4) | 1 |
| r005 | g3 | white diagonal (4×4) | 1 |
| r006 | g3 | white diagonal (4×4) | 1 |

---

## Files

```
course-labs/r42-t001/
├── fixture.json       ← literal synthetic fixture (simulated data)
├── pipeline.py        ← schema validator
├── test_pipeline.py   ← unit tests
└── README.md          ← this file
work/R42-T001/          ← coursework evidence only (remove from any later upstream branch)
├── fixture_input.json      ← input
├── validation_result.json  ← expected vs actual result
├── tool_versions.json      ← Python / platform
├── unittest_output.txt     ← verbose test run (25 tests, exit_code=0)
└── mutation_check.txt      ← one expected value deliberately changed → FAILED, exit_code=1
```

---

## How to Run

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. No dependencies to install (standard library only)

# 3. Run the validator (writes work/R42-T001/ evidence at the repo root)
python3 course-labs/r42-t001/pipeline.py

# 4. Run the automated check (exits non-zero if any assertion fails)
python3 -m unittest discover -s course-labs/r42-t001 -p "test_*.py"
```

---

## Result

- Clean fixture: 6 records validated, 0 failures (`expected == actual`, see `work/R42-T001/validation_result.json`).
- Automated check: 25 tests, all pass, offline and CPU-only, < 1 s.
- Deliberately changing one expected value makes the check fail with exit code 1 (`work/R42-T001/mutation_check.txt`).

These are **pipeline-verification results on simulated data**, not evidence of real-world performance on Fashion-MNIST.

## Limits

- Only validates the schema above; it does not load images, train a CNN, or split data. Because nothing is trained, there is no
  train/validation/test data in this component — the tests use the fixture directly.
- The fixture is deterministic (no random generation), so no seed is needed.
- Within a record, the first bad pixel is reported (with its index); other fields of the same record are still checked.
- No real-data extension is included. One would need to record its source, licence, access requirements and split design separately.

---

## Done When

- [x] Valid fixture passes with zero failures
- [x] Missing field → reported with field name + reason
- [x] Malformed numeric / text / tensor data → reported
- [x] Duplicate `record_id` → reported
- [x] Checks are offline, CPU-only, bounded
- [x] Results labelled as pipeline verification only

---

## Dependencies

| Package | Version used |
|---------|-------------|
| Python  | 3.13.18 (standard library only; also ran on 3.9.6) |
| Third-party packages | none |
