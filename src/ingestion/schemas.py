from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DatasetSchema:
    """Base schema with required columns."""
    required_columns: List[str]

# Define schemas for various datasets
TRANSACTION_COLUMNS = [
    "customer_id",
    "transaction_timestamp",
    "merchant_latitude",
    "merchant_longitude",
    "merchant_zip_plus4",
    "transaction_amount",
]

LOCATION_COLUMNS = [
    "device_id",
    "ping_timestamp",
    "latitude",
    "longitude",
]

MEDIA_EXPOSURE_COLUMNS = [
    "geo_segment_id",
    "demographic_bucket_id",
    "impressions_count",
]

DEMOGRAPHIC_COLUMNS = [
    "geo_segment_id",
    "age_group",
    "income_bracket",
    "interests",
]

CAMPAIGN_RESULTS_COLUMNS = [
    "audience_segment_id",
    "conversion_label",
]
