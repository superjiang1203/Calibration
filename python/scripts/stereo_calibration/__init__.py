"""Image-based sensor intrinsics and pair extrinsics, including stereo baseline."""

from .pairwise import calibrate_session

__all__ = ["calibrate_session"]
