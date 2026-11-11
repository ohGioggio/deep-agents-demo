"""What an agent file defines."""
from dataclasses import dataclass, field

READ_ONLY = ("ls", "read_file", "grep", "glob", "write_todos")


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    prompt: str                              # the system prompt (the file's body)
    tools: tuple | None = None               # this agent's tool names; None = all tools
    disallowed: tuple = ()
    model: str | None = None                 # None = the configured chat model
    skills: tuple = ()                       # skills pinned into its context at start
    max_turns: int | None = None             # lowers the per-run model-call cap
    read_only: bool = False                  # permissionMode: plan
    color: str | None = None
    source: str = ""                         # "built-in" or the file it came from
    frontmatter: dict = field(default_factory=dict, compare=False)  # as read, for round-trip
