# Complete pre-refactor project snapshot

This branch preserves the project immediately before the report-aligned refactor.

- Parent code commit: `46d377973bd9ac33a84b9bf8edf7cbabb0f1a902`.
- Source branch: `main`.
- All current project source, tests, documentation, and the local assignment PDF are included.
- The complete current input CSV and all 764 result files are included, even though they are normally ignored by Git.
- Virtual environments, Python caches, operating-system files, and Git internals are excluded.
- `backup-manifest.json` records every payload file's size, SHA-256 checksum, and Git blob ID.
- Existing historical results and provenance are preserved without modification.
- No model parameters were changed and no experiments were rerun to create this snapshot.

Clone or check out this branch in a separate directory to recover the saved version.
Install the packages in `requirements.txt` to recreate its Python environment.
This branch is a recovery snapshot; further refactoring should proceed separately.
