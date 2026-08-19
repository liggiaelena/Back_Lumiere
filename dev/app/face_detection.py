"""Stage 1: locate and zoom the largest face in an RGB image."""

# The implementation remains import-compatible with the original module while
# this module provides the stable, responsibility-based application API.
from app.face_detection_mediapipe import detect_and_zoom_face, detect_face

__all__ = ["detect_and_zoom_face", "detect_face"]
