# Your agents

Each `.md` file here defines a subagent in Claude Code's format: YAML frontmatter
(`name`, `description`, optional `tools`, `model`, `skills`, `maxTurns`,
`permissionMode: plan` for read-only), then the agent's instructions.
Files from a Claude Code `.claude/agents/` folder work as they are.

Files here are loaded when the demo starts; type `/agents` to see them. The agent
can read this folder but not change it. Subfolders are scanned too, and this
README is ignored. See "Subagents" in the project README for details.
