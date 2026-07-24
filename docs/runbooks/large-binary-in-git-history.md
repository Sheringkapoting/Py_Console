# Runbook: large binary blocking a push, or unrelated-history divergence

Written from this repo's actual incident: connecting a long-unpushed
local `main` (with `models/*.onnx` committed as plain blobs) to a GitHub
`origin/main` that had 6 months of separate, unrelated history. Two
problems stacked: a push-blocking file-size limit, and a merge with no
common ancestor. Both are described here since they tend to show up
together — a repo that's been local-only for a while both accumulates
large files nobody thought about and drifts from its remote.

**Every step below that touches git history or force-pushes anything
needs explicit user confirmation before running — every time, not just
the first time.** This is not a fully-automatable procedure.

## 1. Detect before you push

Before pushing a branch with commits that have never been pushed before,
check for oversized blobs — GitHub hard-rejects anything over 100 MiB:

```bash
git rev-list --objects HEAD \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob" && $3+0 > 20000000 {printf "%.1f MiB  %s\n", $3/1024/1024, $4}' \
  | sort -rn
```

(20 MB threshold is a "worth a second look" bar, not the hard 100 MiB
limit — better to see medium-large files too than to be surprised later.)

If the target remote already exists, also check whether local and remote
histories actually share an ancestor before assuming a normal push/pull
will work:

```bash
git remote add origin <url>
git fetch origin
git merge-base main origin/main   # empty output + exit 1 = unrelated histories
```

## 2. Decide — with the user, not for them

This is the point to stop and ask, not guess:

- **Large files**: Git LFS (rewrite history, keep files versioned) vs.
  strip from history entirely (if they're regenerable/vendored, not real
  source) vs. leave them out of this push only (defers the problem).
- **Unrelated histories**: merge with `--allow-unrelated-histories`
  (combine both timelines) vs. push local as a new branch and leave
  `origin/main` untouched vs. force-push local over remote (destructive —
  needs explicit, unambiguous authorization, and ideally a stated reason
  the remote's unique history doesn't matter).

Different answers are correct in different situations — a stale personal
mirror with no unique remote work is a different case from two active
contributors who diverged. Don't default to force-push.

## 3. Git LFS migration (if that's the choice)

```bash
git lfs install
git lfs migrate import --include="<pattern1>,<pattern2>"   # e.g. "models/**,*.task"
```

Scope `--include` to the actual large-file paths. Avoid `--everything`
unless you specifically intend to rewrite every ref in the repo,
including cached remote-tracking branches (`refs/remotes/origin/*`) — it
rewrote those in this incident too. It turned out harmless here only
because the remote-tracking ref didn't actually contain the large blobs
being migrated (so its commit hashes were untouched), but that's worth
verifying rather than assuming:

```bash
git log --oneline -1 main          # should show new commit hashes
git log --oneline -1 origin/main   # verify this ref wasn't unexpectedly altered
```

### The pointer-stub pitfall

After `git lfs migrate import`, the working tree can end up holding tiny
LFS **pointer files** (~130 bytes, plain text starting `version
https://git-lfs.github.com/spec/v1`) instead of the real binary content —
even though the migrate command reports `Checkout: ..., done.`. This
looks like the files got wiped, but the real bytes are safe in
`.git/lfs/objects`. Verify before panicking:

```bash
ls -la <path-to-large-file>       # if the size is ~130 bytes, it's a pointer stub
du -sh .git/lfs/objects            # confirm the real content is stored locally
git lfs checkout                   # re-smudge pointers back to real content (no network needed)
ls -la <path-to-large-file>        # confirm the real size is back
```

## 4. Merging unrelated histories

```bash
git merge origin/main --allow-unrelated-histories --no-commit
```

Files that exist independently on both sides (no shared ancestor blob)
conflict as `add/add`, even if one side is a trivial edit of the other —
there's no 3-way diff to fall back on, so it's a whole-file decision per
conflicting file, not a line-level merge.

**Before resolving blind, gather signal**: line counts, and a targeted
keyword/def-name diff for anything genuinely ambiguous — don't just take
"ours" or "theirs" for every file without looking. In this incident, that
surfaced which side actually had the more complete implementation for
each conflicting file, and confirmed one file's "shared" name was
misleading (see `docs/architecture.md`'s note on `smart_image_organizer`
naming).

```bash
git checkout --ours  -- <file>   # or --theirs, per-file, after comparing
git add <file>
```

Commit with a message that records *why* each side won — that reasoning
is exactly what's hardest to reconstruct later from `git log` alone.

## 5. Push and verify

```bash
git push -u origin main
```

Large LFS uploads can make the push look like it hung or timed out
client-side even after the actual upload finished — a `git push` retry
immediately after typically reports "Everything up-to-date" rather than
re-uploading. Verify explicitly rather than trusting the first attempt's
output:

```bash
git fetch origin
git log --oneline -1 main
git log --oneline -1 origin/main   # hashes should match
```
