# Working in this repository

## The working tree is shared between concurrent sessions

Several agents edit this checkout at the same time. That is normal here, and it
has one consequence that is not obvious and that has cost real work:

**`git commit -- <path>` commits the file, not your edit.** If another session
has uncommitted changes in the same file, they ride into your commit under your
message. Committing by path is necessary in this tree — staging by path alone is
not enough — but it is not sufficient to isolate your own work.

### Before every commit

```sh
git diff --numstat <each path you are about to commit>
```

Compare against the size of the edit you actually made. A five-line docstring
change that reports `36+/22-` is carrying somebody else's work. This one check
has caught it repeatedly; nothing else in the normal flow will.

Also confirm that every file a CI step runs is tracked, because a step and its
script can come apart the same way:

```sh
git ls-files --error-unmatch <path>
```

A step whose script is untracked fails on `main` while passing locally.
`proofs/validation/empirical/metamorphic/build_flags.py` now gates this, but the
check is cheap to run yourself.

### Landing a commit when the tree is dirty with other sessions' work

`git rebase` refuses while other sessions have unstaged changes, and the fixes
are all worse than the problem: `stash` and `--autostash` put someone else's
work somewhere they will not look for it, and `reset --hard` destroys it.

Use a throwaway worktree instead. It touches the shared tree not at all:

```sh
W=/tmp/land-$$
git worktree add -q --detach "$W" origin/main
# copy in ONLY your own files, from your commit or your editor
git --git-dir="$PWD/.git" show <your-commit>:path/to/File.lean > "$W"/path/to/File.lean
cd "$W"
git diff --numstat            # confirm: only your files, only your line counts
git commit -q -F msg.txt -- path/to/File.lean
git push origin HEAD:main
cd - && git worktree remove --force "$W"
```

If a file in your commit also carries another session's unlanded work, take
`origin/main`'s copy in the worktree and re-apply only your own edit to it. Their
work stays uncommitted in the shared tree, which is where they left it.

### If you have already diverged

`git reset --mixed origin/main` moves the branch pointer and the index and
leaves **every working-tree file byte-for-byte untouched** — nothing is lost,
and other sessions' edits remain as unstaged changes. Verify rather than trust
it:

```sh
find proofs/Calibrator -name '*.lean' -exec shasum {} \; | shasum   # before
git reset --mixed origin/main
find proofs/Calibrator -name '*.lean' -exec shasum {} \; | shasum   # must match
```

Never `checkout`, `restore`, `reset --hard` or `stash` tracked files to undo
something. Undo by editing the file.

## When a checker disagrees with the file in front of you

Assume you are looking at two different revisions before you assume the checker
is broken. There are at least three copies of any source here: your working
tree (which may hold another session's unpushed edit), whatever revision your
build or test checkout is pinned to, and `origin/main`. Check all three
explicitly:

```sh
grep -n <thing> path/to/File.lean                    # working tree
git show origin/main:path/to/File.lean | grep -n <thing>
```

A generated table that refuses to run when stale is doing its job; the staleness
is usually in the checkout, not the tool.
