from feast import FeatureService

from features import customer_profile_features, transaction_features

# Feature service used during training (historical retrieval)
training_service = FeatureService(
    name="training_service",
    features=[
        transaction_features,
        customer_profile_features,
    ],
)

# Feature service used during real-time serving
serving_service = FeatureService(
    name="serving_service",
    features=[
        transaction_features,
        customer_profile_features,
    ],
)
