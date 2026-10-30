from abc import ABC, abstractmethod
from common.schemas import FramePacket

class Camera(ABC):
    @abstractmethod
    def start(self) -> None:
        """Initialize and start the camera stream."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop and close the camera stream."""
        pass

    @abstractmethod
    def read(self) -> FramePacket:
        """Acquire a single synchronized frame packet. Block until available."""
        pass
