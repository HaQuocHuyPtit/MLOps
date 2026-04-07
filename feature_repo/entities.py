from feast import Entity, ValueType

customer = Entity(
    name="customer_id",
    value_type=ValueType.STRING,
    description="Unique customer identifier",
)
