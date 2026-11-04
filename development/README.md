# Development

The roadmap owns the specification. `plan.json` indexes assignable leaves by dotted number and stable ID with explicit dependencies; `workflow.json` preserves workflow tasks outside the product hierarchy. Each leaf also embeds its ordered ancestor contracts, code examples and source URLs. A plan entry is not an active task: its dispatch_ready flag stays false until a coordinator creates and validates a concrete packet. No unattended round starts merely because these files exist.

## Start a round

1. Read the applicable AGENTS.md and the task's full parent feature in ROADMAP.html.
2. The coordinator keeps the claim and evidence in `tasks/<id>.md` using TASK.template.md, then sends one plain-text paragraph from TASK.template.txt as `tasks/<id>.txt`. Embed the relevant current behavior, accepted parent contract, exact snapshot, writable files, dependency receipts, acceptance and finite budget; distinguish current API from proposed syntax. Workers do not fill the coordinator form or reconstruct the conversation.
3. Readiness means every hard dependency has accepted evidence, not that an earlier milestone is complete. Prefer ready work that removes shared bottlenecks; split a leaf when its files, acceptance cases or budget cannot be bounded.
4. Reserve files and hardware before dispatch. One writer per file; the coordinator owns shared exports, settings, IR schemas, dependency manifests and aggregate status. Independent backend files can run in parallel after the common contract.
5. Run a three-task pilot before raising concurrency. Increase only while integration and review keep up. Do not keep adding workers behind a shared bottleneck.

## Implement and verify

- Refresh the primary documentation, upstream source and papers relevant to the assigned slice; record versions, licenses and changed assumptions, reading methods beyond abstracts where research is needed. A small local repair does not require a new survey of unrelated papers.
- Define each numerical case’s input family, dtype, target, independent oracle, metric and justified threshold before optimization; never weaken tolerances merely to pass.
- Through library M4 use only the Mac mini M4 Pro, silo Intel CPU/iGPU and RTX 5080, crimson RTX 3090 GPU 1, and pedro-cvlab RTX 4090; qualify discovery and execution separately.

- Start with the smallest independent acceptance slice. Usually a normal case, a relevant boundary and a failure suffice to expose the intended behavior; that is a heuristic, not a fixed test count.
- Implement a vertical slice. Then expand meaningful Hypothesis properties, regression cases and integration around actual risks. Do not mirror implementation logic in the oracle.
- A research task delivers a sourced recommendation and an executable next step. A contract task delivers signatures, invariants, failures, ownership and acceptance. A build task delivers behavior. A verification task delivers scoped evidence.
- Use a separate reviewer for semantics, numerical behavior and lifetime/concurrency. Add another reviewer when numerical guarantees, unsafe boundaries or cross-backend changes warrant it. Review with the contract, diff and evidence.
- Run focused checks during development; affected integration before acceptance; broad hardware/performance campaigns at release or milestone gates. Use Mainboard for environments, timeouts, reservations and evidence.
- Compilation, execution correctness, repeatability, accuracy and performance are separate results. A coverage percentage describes its recorded denominator and exclusions, not hardware correctness.
- Keep result data in the Mainboard lake. Task files contain commands, receipts and interpretations, not datasets, secrets or large logs.

## Integrate locally

The owner permits local integration after checks and separately authorized the first local commits. This one-time exception does **not** authorize later commits, pushes, tags, releases, deployment or posting. Keep automatic commit/checkpoint jobs disabled.

The integrator checks the baseline, applies only the task's owned edits, resolves shared-file changes and reruns affected checks. Preserve a reversible patch or Mainboard source snapshot without creating a commit. Never stash, hard reset or clean someone else's work. Separate worktrees are optional only when a valid source baseline exists; verify the baseline before creating one.

Use HANDOFF.template.md to record the accepted result. The coordinator updates the roadmap checkbox and the dependency-index status together. A checked Decision means agreement; it never means its implementation shipped. Preserve historical completion and attach a new verification task if its evidence is stale.

## Block or fail

After two materially different failed approaches, or the task budget, stop the failing task and fill BLOCKER.template.md. Stop immediately for a semantic contradiction, exhausted spend cap, data loss risk or approval boundary.

- Missing dependency: name the exact task and missing output; schedule another ready leaf.
- Missing hardware or credentials: retain compile-only evidence and queue the required test. Never replace a required hardware run with a mock pass.
- Defect: retain a minimal reproducer and source/environment identities; assign the repair.
- Unknown design: state the alternatives, recommendation and smallest experiment.
- Owner decision: ask only when it changes public semantics, license/business scope, expenditure or authority. Routine implementation choices follow the accepted contract.

A timeout, skip or expected failure is not success. Never widen a tolerance, suppress an error or mark a requirement optional merely to pass. Requeue only after a named unblocking condition changes. Retry transient infrastructure failures once when safe and within budget; uncertain remote publication must be reconciled before retrying.

## Task state

- `[ ]` Open. No writer owns it.
- `[~]` Active. An owner and bounded budget are recorded.
- `[!]` Blocked. A reproducer, reason and unblock condition exist.
- `[x]` Accepted. Required evidence and review are recorded.

Only the coordinator changes claims and aggregate state. Workers return brief progress or blocker paragraphs with changed files, commands and evidence receipts; the coordinator retains these with the task record. Heartbeats are evidence of progress, not permission to reclaim a live worker's files; contact or stop the worker before reassignment.

The `development/examples/*.txt` briefs are blocked examples, not active assignments. Missing snapshots, reservations, decisions or budgets must be resolved before dispatch. Keep secrets, datasets and large logs out of both the coordinator record and worker paragraph.

The workflow borrows preparation, narrow work allocation and independent review from [Bun's rewrite account](https://bun.com/blog/bun-in-rust). Its throughput and reviewer ratios are not assumed to transfer to a new kernel language.

## Claude Code dynamic workflow

Use `WORKFLOW.template.txt` as the coordinator request and `tasks/<id>.txt` as each worker's complete paragraph. Ask Claude Code to build a dynamic workflow from that request; inspect and save the resulting orchestration through `/workflows`. The workflow must validate dependency receipts and file reservations before dispatch, limit the pilot to three workers, review results independently, integrate locally, and schedule a new ready leaf when another blocks. A workflow script is orchestration, not a numerical specification.

Pin the Claude Code version and verify which project instructions actually load; an ancestor CLAUDE.md can take precedence over AGENTS.md. In noninteractive mode, plain stdin is supported, but the `ultracode` keyword in `-p` does not itself opt into a workflow. Do not use bypass permissions or `--bare` as a shortcut: bare mode omits project context and changes authentication requirements. Retain the owner's existing login and explicit release and spend boundaries.

Read the official [dynamic workflow](https://code.claude.com/docs/en/workflows), [noninteractive](https://code.claude.com/docs/en/headless), and [project instruction](https://code.claude.com/docs/en/memory#agents-md) documentation before implementing the dispatcher; this planning pass launches no Claude Code campaign.
