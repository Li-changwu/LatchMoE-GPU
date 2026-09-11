from __future__ import annotations


class LatchMoEError(RuntimeError):
    """Base error for failures that must be retained in experiment logs."""


class ManifestHashError(LatchMoEError):
    def __init__(self, *, expected: str, actual: str):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"manifest SHA-256 mismatch: expected={expected}, actual={actual}"
        )


class ManifestValidationError(LatchMoEError):
    pass


class UnsupportedVllmVersionError(LatchMoEError):
    def __init__(self, *, expected: str, actual: str):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"unsupported vLLM version: expected={expected}, actual={actual}"
        )


class LayoutMismatchError(LatchMoEError):
    pass


class ActiveExpertCapacityError(LatchMoEError):
    pass


class IllegalSlotTransitionError(LatchMoEError):
    pass


class NoEvictableSlotError(LatchMoEError):
    pass


class StaleMappingError(LatchMoEError):
    pass


class StableAddressError(LatchMoEError):
    pass


class StagingDuringCaptureError(LatchMoEError):
    pass


class PairIntegrityError(LatchMoEError):
    pass


class ResidencyBudgetError(LatchMoEError):
    """The requested residency cannot satisfy the explicit HBM contract."""


class PlanValidationError(LatchMoEError):
    """A residency plan is malformed or does not match its model."""


class IdentityLockError(LatchMoEError):
    """The model/vLLM identity lock cannot be verified."""


class NativeCombineError(LatchMoEError):
    """The locked vLLM seam cannot perform the required single combine."""


class CapabilityError(LatchMoEError):
    """The requested model/runtime tuple is outside the qualified matrix."""
