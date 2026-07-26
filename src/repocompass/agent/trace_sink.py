"""Persistence boundary for RepoCompass domain traces.

The Runner creates an in-memory ``RunTrace``. A sink receives that completed
trace and stores it without interpreting graph nodes or changing event data.
"""

from pathlib import Path

from repocompass.agent.trace import RunTrace


class JsonlTraceSink:
    """Persist a completed :class:`RunTrace` as newline-delimited JSON.

    This concrete M0 sink deliberately has no database abstraction, buffering
    policy, or LangSmith integration. Each call writes one completed run to one
    caller-selected file.
    """

    def write(self, trace: RunTrace, destination: Path) -> Path:
        """Write ``trace`` to a new UTF-8 JSONL file.

        Args:
            trace: Completed in-memory trace. Events must be written in their
                existing sequence order without mutation.
            destination: Exact output file selected by the caller. Its parent
                directory may not exist yet.

        Returns:
            The resolved path of the newly created JSONL file.

        Raises:
            FileExistsError: If ``destination`` already exists. Existing traces
                must never be overwritten silently.
            OSError: If a directory or file cannot be created or written.

        Side effects:
            Creates missing parent directories and exactly one trace file. It
            must not call the graph, modify ``trace``, or contact LangSmith.

        Storage contract:
            Write exactly one ``TraceEvent.model_dump_json()`` value per line,
            followed by ``"\\n"``. Do not wrap the events in a JSON array.
        """

        normalized_path = destination.expanduser().resolve()
        normalized_path.parent.mkdir(parents=True, exist_ok=True)

        with normalized_path.open("x", encoding="utf-8") as file:
            for event in trace.events:
                file.write(event.model_dump_json())
                file.write("\n")

        return normalized_path
