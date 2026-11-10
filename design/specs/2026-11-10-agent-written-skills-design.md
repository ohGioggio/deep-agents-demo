# Agent-written skills with approval (pieces 1 and 2)

Status: approved in chat on 2026-11-10 ("its good, start"). Pieces 3 (marketplace)
and 4 (tools registry) and the risky-tool guardrail come later.

## Intent

Let the agent create, update, remove, import and restore skills, but only when the
user asks, and only after the user approves each change. Skills must always be in
the format Deep Agents loads, never contain secrets, and keep every version.

## Decided by the user

- Skills are per user now; a global "marketplace" scope comes in piece 3. The same
  name may exist in both scopes, and every operation states its scope.
- Changes happen only on request. Before approving, the user sees a summary, and on
  request the diff or the full file. No editing of drafts in the prompt.
- Storage is SQLite. Skills are versioned. A skill may include extra files.
- "Load" includes importing a skill from a file or folder the user points at.
- The agent never writes tools (code) for now.

## Design

**Approval step.** Gated tools are paused by LangChain's `HumanInTheLoopMiddleware`
through Deep Agents' `interrupt_on` (decisions: approve, or reject with a reason).
Subagents inherit it, so delegation cannot skip approval. The chat loop catches the
interrupt, renders each request, and resumes with the decisions. Validation runs
before the user is asked: a change that fails it is rejected automatically with the
errors, so the agent can fix it. With `--prompt` there is no human, so every change
is rejected automatically.

**Prompt shown to the user** (summary first; `d` diff, `f` full files):

    Skill change requested: UPDATE "street-works-checklist" (user: demo-user), v2 -> v3
      Lists the data a worksite record needs, from the data architecture doc.
      Changes: +12 lines, -3 lines · files: SKILL.md, template.md
      Checks: format OK · no secrets found
    [a]pprove  [r]eject (with a reason)  [d]iff  [f]ull file >

**Tools** (feature `skill_management`, user scope only for now):
`create_skill`, `update_skill`, `remove_skill`, `import_skill`, `restore_skill` are
gated; `skill_history` is read-only. The prompt tells the agent to use them only
when asked and not to retry a rejected change unless asked.

**Format and safety checks.** Name and frontmatter use Deep Agents' own validators
(`_validate_skill_name`, `_parse_skill_metadata`), so a saved skill is always
loadable. Extra file paths must be relative with no `..`. Every file is scanned with
`memory_policy.find_secrets`. Size limits follow Deep Agents' `MAX_SKILL_FILE_SIZE`.

**Storage.** `data/skills.sqlite`: a `skills` table (scope, owner, name, current
version, removed flag) and a `skill_versions` table (all files of each version as
JSON, action, time). Remove and restore add versions, so history is never lost.

**Loading.** The current version of each of the user's skills is written to
`demo_workspace/user-skills/<user>/`, added as a skills source after the seeded
`skills/` folder (so a user skill overrides a seeded one with the same name). The
folder is rewritten from the database on start and after each approved change;
skills are reloaded every turn, so a change is visible from the next message.

**No bypass.** File-tool writes to every skill folder (`skills/`, `sandbox-skills/`,
`user-skills/`) are denied with Deep Agents' `FilesystemPermission`, so the gated
tools are the only way to change a skill.

## Calls made by Claude (check these)

- Same-name precedence: user skills override seeded and, later, marketplace skills.
- The database is the source of truth; the `user-skills/` folder is regenerated and
  hand edits there are overwritten.
- Skills reload on every turn (cheap: local files) instead of tracking changes.
- `import_skill` reads text files only (no binaries) and caps total size.
- Secret rule for skills (changed during the build): concrete credential formats
  (key prefixes, private keys, credentials in URLs, card numbers, random tokens) are
  rejected; wording such as "the token is rotated every 90 days" is only a warning in
  the summary, because skills legitimately explain keys and tokens. Memory keeps the
  strict rule.

## Left to the builder

Exact table columns, error wording, how the diff is formatted.
