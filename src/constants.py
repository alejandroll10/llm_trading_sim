"""Centralized constants for the simulation framework.

This module contains system-wide constants used across the codebase to ensure
consistency and avoid duplication.
"""

# Floating point comparison tolerance
# Used for validating commitments, releases, and other monetary calculations
# to account for floating point precision errors
FLOAT_TOLERANCE = 1e-5

# Cash position matching tolerance
# Used for verifying cash positions match payment history
# Larger tolerance (1 cent) for cash matching since we're checking aggregates
CASH_MATCHING_TOLERANCE = 0.01
