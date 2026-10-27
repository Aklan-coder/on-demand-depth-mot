"""On-demand monocular depth for multi-object tracking (CSCE 585 project).

When is depth worth computing? A ByteTrack tracker that invokes Depth Anything V2 only
when association is ambiguous, evaluated against no / always / fixed-rate / pseudo depth.
"""
__version__ = "0.1.0"
