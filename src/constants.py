"""Centralized constants for the simulation framework.

This module contains system-wide constants used across the codebase to ensure
consistency and avoid duplication.
"""

# Floating point comparison tolerance
# Used for validating commitments, releases, and other monetary calculations
# to account for floating point precision errors
FLOAT_TOLERANCE = 1e-5
