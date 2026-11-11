"""The documents feature: the search tool and its prompt."""
from deep_agents_demo.core.feature import Feature
from deep_agents_demo.features.documents.index import DocumentIndex


def documents_prompt(sources):
    """Tell the agent what the document index holds, so it knows when to search it."""
    if not sources:
        return ("Use search_docs for questions about the user's documents and cite source "
                "names. ")
    shown = ", ".join(sources[:30]) + (f" and {len(sources) - 30} more" if len(sources) > 30 else "")
    return (f"The user's indexed documents are: {shown}. For any question these documents "
            "could answer, call search_docs first, even if the user does not mention "
            "documents, and base the answer on the passages, citing source names. If nothing "
            "relevant comes back, say so before using general knowledge. These documents "
            "are only reachable through search_docs; they are not files you can read. ")


def documents_feature(index: DocumentIndex) -> Feature:
    """Document search: the tool, for the agent and the researcher, and when to use it."""
    search_docs = index.tool()
    return Feature(name="documents", prompt=documents_prompt(index.sources()),
                   tools=(search_docs,), lookups=(search_docs,))
