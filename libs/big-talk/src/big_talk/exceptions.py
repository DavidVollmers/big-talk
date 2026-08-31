from big_talk.message import ToolResult


class SuspensionError(Exception):
    """Raised by a Middleware to suspend the agentic loop.

    The exception propagates to the caller, who is responsible for handling the
    suspension — typically by persisting state and resuming the loop later.

    Args:
        details: Optional context about why the loop was suspended (e.g. a
            checkpoint object or status message). Available as ``self.details``.
        children: Optional map of parent IDs to child suspension errors, used
            when multiple tools suspended in the same iteration.
        message: Optional human-readable message passed to ``Exception.__init__``.

    Example::

        raise SuspensionError({"checkpoint": state}, message="human approval required")
    """

    def __init__(self, details: object = None, children: dict[str, list['SuspensionError']] = None,
                 message: str = None):
        self.details = details
        self.children = children
        super().__init__(message)

    def __str__(self):
        if self.details is None:
            return super().__str__()
        return f"LoopSuspensionError: The Agent Loop was Suspended with the following details: {self.details}"


class BatchSuspendedException(Exception):
    """
    Raised when a batch of tools is interrupted by a HITL suspension.
    Carries both the tools that require approval and the tools that
    already successfully completed in the parallel batch.
    """

    def __init__(self, suspensions: dict[str, list[SuspensionError]], partial_results: dict[str, list[ToolResult]]):
        self.suspensions = suspensions
        self.partial_results = partial_results
        super().__init__(
            f"Batch suspended: {len(suspensions)} pending approvals, {len(partial_results)} completed results.")
