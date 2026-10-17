# Tuning log

One entry per eval run that informed a decision, newest first. Record the change, the
clips used, and the numbers from `soundthing eval` (fine/coarse F1, FP/min) for both
baseline and conditioned. Note what was kept or reverted and why.

Entries titled `promote <run_id>` are appended automatically by `promote` (format above,
holdout shown as pass only). Keep the manual format block below for hand-run evals.

Format:

```
## YYYY-MM-DD — <what changed>
Clips: <which / how many>    Config: <relevant settings>
Baseline:    F1 fine x.xx / coarse x.xx, FP/min x.x
Conditioned: F1 fine x.xx / coarse x.xx, FP/min x.x
Outcome: <kept / reverted, and why>
```

_No runs yet._
