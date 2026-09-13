"""Port for synchronizing physically attached payloads after physics steps."""

from abc import ABC, abstractmethod


class IAttachmentSynchronizer(ABC):
    """Keep attached payload bodies aligned with their owning robot."""

    @abstractmethod
    def sync_attachments(self) -> None:
        """Apply attachment constraints after one physics step."""
        raise NotImplementedError
