"""The files feature: the agent delivers documents, data and pages as files in the chat.

The agent writes a file with its usual tools (``write_file``, ``edit_file``, or code in
a sandbox), then calls ``send_file``. The tool reads the file through the agent's
backend (so local files and sandboxes behave the same), checks it, and keeps it as
a new version in the output store, where the agent can't change it. The model never
chooses whose chat a file lands in: the user comes from code, the chat from the
run's thread.
"""
import json

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import SystemMessage
from langgraph.config import get_config

from deep_agents_demo.core.feature import Feature
from deep_agents_demo.features.files.validate import clean_name, describe, validate
from deep_agents_demo.persistence.outputs import OutputStore

MODES = ("auto", "always", "never")

FIRST_SENTENCE = ("When the user asks for a document, report, guide, plan, dataset, page or any "
                  "content they will keep or reuse, write it to a file and deliver it with send_file.")
ALWAYS_FIRST_SENTENCE = ("Use a file for every answer longer than a few lines: write it to a file and "
                         "deliver it with send_file.")
REST = (" Choose the format from the content: .md for documents, .csv for tabular data, .html for "
        "pages, .tex when they ask for LaTeX, .svg for simple graphics, .txt for plain notes. Answer "
        "in chat for questions, explanations and short replies; as a guide, anything with a title "
        "and sections, or longer than about a page, is a file. Use short kebab-case names. After "
        "sending a file, reply with one or two sentences about it and never repeat its content. "
        "When asked to change a file you sent, edit that file and send it again; don't start a new "
        "one. If the user says \"answer here\" or \"as a file\", do that. ")


def _chat(config, user: str) -> tuple[str | None, str | None]:
    """(chat name, None), or (None, why not): the run must be one of ``user``'s chats."""
    raw = (config or {}).get("configurable", {}).get("thread_id")
    try:
        owner, chat = json.loads(raw)
    except (TypeError, ValueError):
        return None, "send_file works only in a chat (this run isn't one); answer in the chat instead."
    if owner != user or not isinstance(chat, str):
        return None, "send_file can only deliver files to the current user's chat."
    return chat, None


def make_send_file(backend, outputs: OutputStore, user: str):
    @tool(response_format="content_and_artifact")
    def send_file(path: str, runtime: ToolRuntime, caption: str = "") -> tuple[str, dict | None]:
        """Deliver a file you wrote (with write_file, edit_file or code) to the user in the
        chat, as the result. path: the file as you wrote it. caption: one short line about
        it (optional). Sending the same name again delivers a new version."""
        chat, problem = _chat(runtime.config, user)
        if problem:
            return problem, None
        try:
            name = clean_name(path)
        except ValueError as error:
            return str(error), None
        [found] = backend.download_files([path])
        if found.error or found.content is None:
            return (f"Can't read {path}: {found.error or 'no content'}. Write the file first "
                    "(write_file), then send it."), None
        problem = validate(name, found.content)
        if problem:
            return f"Not sent: {problem}.", None
        saved = outputs.save(user, chat, name, found.content, caption.strip()[:300])
        result = {"name": saved.name, "version": saved.version, "bytes": saved.bytes,
                  "kind": saved.kind, "details": describe(saved.name, found.content)}
        return json.dumps(result), {"saved": True, **saved.to_dict()}

    return send_file


class DeliveredFilesNote(AgentMiddleware):
    """Tell the model which files this chat already has, on every model call, so "the
    document" has a referent after a resume. Nothing is written to the conversation."""

    def __init__(self, outputs: OutputStore, user: str):
        super().__init__()
        self.outputs, self.user = outputs, user

    def _with_files(self, request):
        try:
            config = get_config()
        except RuntimeError:  # not inside a run
            return request
        chat, problem = _chat(config, self.user)
        files = [] if problem else self.outputs.list(self.user, chat)
        if not files:
            return request
        note = ("\n\nFiles delivered in this chat: "
                + ", ".join(f"{f.name} (v{f.version})" for f in files) + ".")
        system = request.system_message
        if system is None:
            return request.override(system_message=SystemMessage(content=note.lstrip()))
        if isinstance(system.content, str):
            content = system.content + note
        else:
            content = [*system.content, {"type": "text", "text": note}]
        return request.override(system_message=SystemMessage(content=content))

    def wrap_model_call(self, request, handler):
        return handler(self._with_files(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._with_files(request))


def files_feature(backend, outputs: OutputStore, user: str, mode: str = "auto") -> Feature | None:
    """The feature for ``FILE_OUTPUT``: None when it is "never"."""
    if mode == "never":
        return None
    prompt = (ALWAYS_FIRST_SENTENCE if mode == "always" else FIRST_SENTENCE) + REST
    return Feature(name="files", prompt=prompt, tools=(make_send_file(backend, outputs, user),),
                   middleware=(DeliveredFilesNote(outputs, user),))
