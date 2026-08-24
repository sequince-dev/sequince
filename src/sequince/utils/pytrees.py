from __future__ import annotations

import optree

OPTREE_NAMESPACE = "sequince"


def pytree_dataclass(*, data_fields, meta_fields=(), drop_fields=()):
    """Register a dataclass as an Optree and, when available, JAX pytree."""
    data_fields = tuple(data_fields)
    meta_fields = tuple(meta_fields)
    drop_fields = tuple(drop_fields)

    def register(cls):
        def flatten(instance):
            children = tuple(getattr(instance, name) for name in data_fields)
            metadata = tuple(getattr(instance, name) for name in meta_fields)
            return children, metadata

        def unflatten(metadata, children):
            values = dict(zip(data_fields, children, strict=True))
            values.update(zip(meta_fields, metadata, strict=True))
            return cls(**values)

        optree.register_pytree_node(
            cls,
            flatten,
            unflatten,
            namespace=OPTREE_NAMESPACE,
        )

        try:
            import jax
        except ModuleNotFoundError as exc:
            if exc.name not in {"jax", "jaxlib"}:
                raise
        else:
            jax.tree_util.register_dataclass(
                cls,
                data_fields=data_fields,
                meta_fields=meta_fields,
                drop_fields=drop_fields,
            )
        return cls

    return register
