"""The memory features: Mem0 memory and Deep Agents' file memory."""
from deep_agents_demo.core.feature import Feature
from deep_agents_demo.features.memory.recall import AutoRecallMiddleware
from deep_agents_demo.features.memory.service import MemoryService
from deep_agents_demo.features.memory.tools import memory_tools

_RECALL_PROMPT = {
    "tool": "You can't see the user's saved preferences unless you call recall. Call it "
            "first whenever the answer could depend on who the user is or what they prefer: "
            "questions about the user, and recommendations or choices a personal preference "
            "could change (tools, styles, units, budgets, colours). ",
    "auto": "Relevant memories are added automatically under 'Recalled memories'; use "
            "them when they apply. There is no recall tool. ",
    "both": "Relevant memories are added automatically under 'Recalled memories'; if they "
            "don't cover a preference the answer depends on, call recall. ",
}


def memory_feature(service: MemoryService, recall_mode="auto") -> Feature:
    """Mem0 long-term memory for one user: tools, rules and automatic recall."""
    ids_from = "list_memories" if recall_mode == "auto" else "list_memories or recall"
    prompt = (_RECALL_PROMPT[recall_mode]
              + "Use remember only when explicitly asked to save a fact. Never store secrets "
              "(passwords, API keys, tokens, card numbers); remember refuses them, so tell the "
              "user to keep them in a password manager instead. Use list_memories when asked "
              f"what you remember, and forget with an id from {ids_from} when asked to forget "
              "or correct something. Mem0 is the source of long-term user preferences; do "
              "not duplicate them in AGENTS.md. ")
    return Feature(
        name="memory",
        prompt=prompt,
        tools=tuple(memory_tools(service, recall_tool=recall_mode != "auto")),
        middleware=(AutoRecallMiddleware(service),) if recall_mode != "tool" else (),
    )


def file_memory_feature(skills_prefix="") -> Feature:
    """Without Mem0: preferences live in AGENTS.md, loaded by Deep Agents' file memory."""
    path = f"{skills_prefix}/memories/AGENTS.md"
    return Feature(name="file-memory",
                   prompt=f"Save explicitly requested preferences in {path}. ",
                   memory_sources=(path,))
