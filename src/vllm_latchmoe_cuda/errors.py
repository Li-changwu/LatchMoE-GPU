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
        super().__init__(f"unsupported vLLM version: expected={expected}, actual={actual}")


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

