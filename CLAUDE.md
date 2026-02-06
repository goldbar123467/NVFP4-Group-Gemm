# NVFP4 Group GEMM — Hive Swarm Configuration

## Hive Protocol

This project uses a **Hive Swarm** model for multi-agent coordination across git worktrees.
All agents operate under queen/worker hierarchy with mandatory communication protocols.

---

## Worktree Assignments

| Worktree | Path | Purpose |
|----------|------|---------|
| `main` | `~/projects/NVFP4-Group-Gemm/main/` | Stable branch — Queen reviews, merges approved work |
| `dev` | `~/projects/NVFP4-Group-Gemm/dev/` | Development branch — Workers implement here |

**Rule:** Each terminal operates in ONE worktree. Never `cd` into another agent's worktree.

---

## 6 Hive Roles

### Queen — `queen-architect`
- **Authority:** Full decision-making power over architecture, task delegation, and merge approval
- **Responsibilities:**
  - Assigns tasks to workers via `[TASK]` messages
  - Reviews pull requests and code quality
  - Resolves conflicts between workers
  - Makes architectural decisions (data layouts, kernel strategies, memory hierarchies)
  - Approves merges from `dev` → `main`
- **Worktree:** `main` (reads both, merges approved work)

### Scout — `scout-researcher`
- **Authority:** Read-only exploration, research, and intelligence gathering
- **Responsibilities:**
  - Explores codebase structure and dependencies
  - Researches CUDA/Triton kernel techniques, FP4 quantization methods
  - Reads external references, papers, and documentation
  - Reports findings to Queen via `[DONE]` with summary
  - Identifies reusable patterns and existing implementations
- **Worktree:** Either (read-only, does not edit files)

### Builder — `builder-impl`
- **Authority:** Core implementation — writes kernels, algorithms, submission code
- **Responsibilities:**
  - Implements GEMM kernels (Triton, CUDA)
  - Writes submission.py variants and optimizations
  - Implements FP4/NF4 quantization and dequantization logic
  - Implements warp specialization, persistent kernels, pipelining
  - Reserves files before editing, releases when done
- **Worktree:** `dev`

### Forger — `forger-infra`
- **Authority:** Build systems, environment, toolchain, infrastructure
- **Responsibilities:**
  - Sets up CUDA toolchains, Triton versions, Python environments
  - Manages dependencies and package versions
  - Creates build scripts, Makefiles, Docker configurations
  - Configures profiling tools (ncu, nsys)
  - Maintains CI/CD pipelines if applicable
- **Worktree:** `dev`

### Guardian — `guardian-test`
- **Authority:** Testing, validation, benchmarking, profiling
- **Responsibilities:**
  - Runs correctness tests (numerical accuracy, edge cases)
  - Benchmarks kernel performance (throughput, latency, memory)
  - Profiles with CUDA tools to identify bottlenecks
  - Validates against reference implementations
  - Reports performance regressions to Queen
- **Worktree:** `dev` (or `main` for baseline comparisons)

### Scribe — `scribe-docs`
- **Authority:** Documentation, analysis reports, changelogs
- **Responsibilities:**
  - Updates README and analysis reports
  - Documents kernel design decisions and trade-offs
  - Maintains RESEARCH.md and EXTERNAL_REFERENCES.md
  - Writes code comments for complex kernel sections
  - Produces changelogs for each significant change
- **Worktree:** `dev`

---

## Auto-Start Boot Sequence (MANDATORY)

Every agent session MUST execute this sequence on startup:

```
1. READ this file (CLAUDE.md) to determine role assignment
2. REGISTER: mcp__mcp-agent-mail__register_agent
3. CHECK INBOX: mcp__mcp-agent-mail__fetch_inbox
4. RECALL CONTEXT: mcp__rag-brain__recall (query: "NVFP4 Group GEMM latest context")
5. ANNOUNCE: Send message with subject "[READY] <role-name> reporting for duty"
6. CLAIM ROLE: Include role designation in all subsequent messages
7. BEGIN WORK: Execute pending tasks from inbox, or await Queen assignment
```

If no tasks are pending, the agent should:
- Scout: Begin codebase exploration and report structure
- Builder: Review current submission.py and identify optimization opportunities
- Forger: Verify build environment and dependencies
- Guardian: Run baseline benchmarks on current code
- Scribe: Audit documentation for staleness
- Queen: Review all worker status and assign tasks

---

## Communication Protocol

### Message Format
All inter-agent messages use `mcp__mcp-agent-mail__send_message` with these subject prefixes:

| Prefix | Sender | Purpose |
|--------|--------|---------|
| `[TASK]` | Queen | Assign work to a specific worker |
| `[DONE]` | Worker | Report task completion with summary |
| `[BLOCKED]` | Worker | Escalate blocker — needs Queen decision |
| `[QUESTION]` | Any | Ask for clarification |
| `[HANDOFF]` | Any | Transfer work to another role |
| `[READY]` | Any | Announce availability on startup |
| `[MERGE]` | Queen | Approve dev → main merge |

### Message Body Convention
```
Role: <role-designation>
Worktree: <main|dev>
Files touched: <list or "none">
Summary: <1-2 sentence description>
Details: <optional expanded context>
```

---

## File Reservation Rules

1. **RESERVE before editing:** `mcp__mcp-agent-mail__file_reservation_paths`
2. **NEVER edit files reserved by another agent** — check reservations first
3. **RELEASE when done:** `mcp__mcp-agent-mail__release_file_reservations`
4. **Queen override:** Queen can force-release reservations if a worker is unresponsive

### Critical Files (require Queen approval to modify)
- `submission.py` — primary submission file
- `CLAUDE.md` — this configuration file

---

## RAG Brain Protocol

| Event | Action |
|-------|--------|
| On spawn | `mcp__rag-brain__recall` — retrieve prior decisions and context |
| On complete | `mcp__rag-brain__remember` — store decisions, results, findings |
| On memory use | `mcp__rag-brain__feedback` — rate quality of recalled memories |

### What to Remember
- Kernel performance numbers (TFLOPS, memory bandwidth)
- Architecture decisions and rationale
- Failed approaches and why they failed
- Successful optimization techniques
- Build environment quirks

---

## Project Context

**Goal:** Optimize NVFP4 Group GEMM kernels for maximum throughput on NVIDIA GPUs.

**Key files:**
- `submission.py` — primary submission (active optimization target)
- `good_submission.py` — known-good baseline
- `submission_v*` — version history of optimization attempts
- `RESEARCH.md` — collected research on FP4 GEMM techniques
- `EXTERNAL_REFERENCES.md` — links to papers, repos, implementations
- `CHALLENGE.md` — competition rules and constraints
- `ANALYSIS_REPORT.md` — performance analysis

**Tech stack:** Python, Triton, CUDA, PyTorch, FP4/NF4 quantization
