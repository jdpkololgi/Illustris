# SCIENCE_LOG.md — shared brain: Claude Desktop (science) ⇄ Claude Code (NERSC)

The boundary object that keeps **science discussions** (Claude Desktop, local Mac)
and **agent work** (Claude Code, NERSC) aware of each other. It rides git, so it
reaches every machine.

- NERSC path: `~/TNG/Illustris/SCIENCE_LOG.md`
- Mac path:   `~/Developer/Illustris/SCIENCE_LOG.md`
- Syncs through GitHub (`jdpkololgi/Illustris`).

## How to use this file (both assistants)

1. **Read it at the start of a session** to load current direction before doing
   anything substantive.
2. **Append a short entry** (newest at the top of the Log) when a decision is
   made, a result lands, or direction changes. Don't rewrite history; add.
3. **It's only current after `git pull`.** Pull before reading/editing; commit +
   push after writing. (Stage just this file: `git add SCIENCE_LOG.md`.)
4. Keep entries terse and skimmable. Prune "Open threads" as items close.

Tags: `[science]` = decisions/hypotheses/conclusions from Desktop discussions.
`[code]` = what was run/changed/found, blockers, next actions from NERSC.

Entry shape:

```
### YYYY-MM-DD — [science|code] short title
- What: ...
- Why / decision: ...
- Next: ...
- Refs: files, commits, run dirs
```

## Open threads / current focus

- **NPE on Abacus wedge subvolumes** is the path forward for SBI. Graph-partitioned
  FlowJAX is legacy ("a nightmare") and to be retired when convenient.
- **GraphWeb production VAC** needs a real `sbatch` submit script — the DESI
  graph→features→inference chain currently only runs interactively.
- Jraph regression on wedge/cube caches is active and fine.

## Log (newest first)

### 2026-06-14 — [code] Cross-tool infra + this log
- What: Set up `~/.claude/CLAUDE.md` as cross-repo canon (3-location map, conda
  envs, proven Perlmutter `salloc`/`srun` recipes; production = `sbatch`). Fixed
  the `nersc` skill's `qos.md` (no `gpu_`-prefixed QOS). Added two Claude Code
  skills: `desi-bgs-graph` and `jraph-eval`. Created this SCIENCE_LOG.
- Next: stand up the Desktop⇄NERSC bridge (filesystem MCP + git) so science
  decisions land here automatically.
- Refs: `~/.claude/CLAUDE.md`, `~/.claude/skills/{desi-bgs-graph,jraph-eval}`.

### 2026-06-14 — [science] SBI direction: wedges, not partitions
- What: Abacus-scale inference works via **wedge subvolumes** (graph per RA/Dec/z
  wedge); the graph-partitioning / partitioned-FlowJAX path is abandoned.
- Why / decision: graph partitioning was a nightmare; wedge training has worked
  over the last ~2 months. Future = NPE on the wedge subvolumes.
- Next: when convenient, retire `workflows/sbi/*partitioned*` + partition builders.
- Refs: `workflows/sbi/`, see also Claude Code memory `project-abacus-wedge-npe-direction`.
