from typing import Any, Dict, Optional


class Registry:
    registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        def inner_wrapper(wrapped_class):
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs):
        model = cls.registry[name]
        model = model(**kwargs)
        return model
