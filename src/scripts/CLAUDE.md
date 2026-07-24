# src/scripts/ — shared engine, high blast radius

Two modules here are imported by most of the repo's CLI tools. A change
here isn't scoped to one tool — it's scoped to everything that imports it.
There is no automated test suite yet, so the check below is the only
safety net.

## What's here and who depends on it

- **`common_utils.py`** — `TerminationManager` (ESC-abort), imported by
  **all 10 tools**. Also `format_size`, `ProgressBarHelper`,
  `validate_file_path`, `validate_directory`, `safe_filename`,
  `SimpleTimer` — used more selectively (see `docs/architecture.md` for
  the exact breakdown).
- **`dup_finder_core.py`** — the full duplicate-finder engine (`UnionFind`,
  SHA-256 pass, `MediaHandler` interface, near-match grouping, Rich UI,
  HTML report, `run_workflow`). Currently consumed by **one** tool,
  `find_media_duplicates.py`. `find_duplicates.py` has its own separate,
  not-yet-migrated implementation of the same shape — see
  `docs/architecture.md` for why that migration hasn't happened yet.

## Before changing either file

1. Check `docs/architecture.md`'s "who uses what" table for the actual
   current consumer list — don't assume it matches this file if enough
   time has passed; re-grep for `from common_utils import` /
   `import dup_finder_core` across the repo root if in doubt.
2. After the change, actually run at least one consuming tool in dry-run
   mode end-to-end (not just `python -m py_compile`) — a syntax-clean
   change can still silently break a downstream tool's behavior. This is
   the same practice used when `find_media_duplicates.py` was first built
   on top of `dup_finder_core.py`.
3. If the change is to `TerminationManager` specifically, verify ESC-abort
   still works in at least one tool — it's the one safety mechanism nearly
   every tool in this repo relies on.

## Import pattern (don't change this without a reason)

Every consumer bootstraps the same way, because there's no installed
package (`src/scripts` isn't `pip install -e`'d — see
`docs/architecture.md`):

```python
_scripts_dir = str(Path(__file__).parent / "src" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from common_utils import TerminationManager as _TerminationManager
```

If you're adding a new shared module here, follow this exact pattern in
its consumers rather than introducing a different import mechanism.
