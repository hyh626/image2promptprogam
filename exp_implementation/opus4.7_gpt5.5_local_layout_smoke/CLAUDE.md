# Implementation Agent Context

This prepared folder is currently in the implementation/bootstrap phase.

Read `kickoff-prompt.txt`, then follow `IMPLEMENTATION.md`. Also read
`program.md` for the user-facing research contract and
`EVAL_STORAGE_SCHEMA.md` for canonical eval artifact storage.

If `program.md` says the harness already exists, only `prompt_strategy.py` may
be edited, or the harness must not be modified, treat that as instructions for
the later research/driver phase. During this implementation phase,
`IMPLEMENTATION.md` is the implementation contract.

For eval storage layout and metadata files, `EVAL_STORAGE_SCHEMA.md` wins over
all other docs. Keep `check_eval_storage.py` available and run
`python check_eval_storage.py --root .` after producing eval artifacts.

After the harness is built and handed off, replace this context file with a
copy or symlink to `program.md` before starting the research/driver agent.
