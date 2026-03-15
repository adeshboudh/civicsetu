from enum import StrEnum


class Jurisdiction(StrEnum):
    CENTRAL = "CENTRAL"
    MAHARASHTRA = "MAHARASHTRA"
    KARNATAKA = "KARNATAKA"
    UTTAR_PRADESH = "UTTAR_PRADESH"
    TAMIL_NADU = "TAMIL_NADU"
    UNKNOWN = "UNKNOWN"


class DocType(StrEnum):
    ACT = "ACT"
    RULES = "RULES"
    CIRCULAR = "CIRCULAR"
    ORDER = "ORDER"
    NOTIFICATION = "NOTIFICATION"
    AMENDMENT = "AMENDMENT"


class QueryType(StrEnum):
    FACT_LOOKUP = "fact_lookup"
    CROSS_REFERENCE = "cross_reference"
    CONFLICT_DETECTION = "conflict_detection"
    TEMPORAL = "temporal"
    PENALTY_LOOKUP = "penalty_lookup"
    UNKNOWN = "unknown"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChunkStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    PENDING = "pending"
