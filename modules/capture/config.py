from pydantic import BaseModel, model_validator
from typing import Optional
from common.schemas import CameraIntrinsics

class CameraConfig(BaseModel):
    width: int = 640
    height: int = 480
    fps: int = 30
    fx: float = 615.0
    fy: float = 615.0
    cx: Optional[float] = None
    cy: Optional[float] = None

    @model_validator(mode='after')
    def set_default_centers(self) -> 'CameraConfig':
        if self.cx is None:
            self.cx = float(self.width) / 2.0
        if self.cy is None:
            self.cy = float(self.height) / 2.0
        return self

    def to_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height
        )
