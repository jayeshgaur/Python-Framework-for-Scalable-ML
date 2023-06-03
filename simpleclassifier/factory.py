from abc import ABC

from simpleclassifier.base import (Classifier, SplitterDataset, Splitter)


class BaseFactory(ABC):
    registry = None

    def __init_subclass__(cls):
        cls.registry = {}

    @classmethod
    def register(cls, name: str):

        def inner_wrapper(wrapped_class):
            if not issubclass(wrapped_class, cls.get_registered_class_type()):
                raise TypeError(
                    f"{wrapped_class.__name__} is not a subclass of "
                    f"{cls.get_registered_class_type().__name__} ")
            if name in cls.registry:
                raise ValueError(
                    f"{wrapped_class.__name__} {name} already exists")
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_registered_class_type(cls):
        raise NotImplementedError(
            "Subclasses must implement get_registered_class_type method")

    @classmethod
    def create_instance(cls, name: str, **kwargs):
        if name not in cls.registry:
            raise ValueError(
                f"No {cls.__name__.lower()} registered for {name}, registered "
                f"{cls.__name__.lower()}s are: {list(cls.registry.keys())}")
        class_ = cls.registry[name]
        instance = class_(**kwargs)
        return instance


class ClassifierFactory(BaseFactory):

    @classmethod
    def get_registered_class_type(cls):
        return Classifier


class SplitterDatasetFactory(BaseFactory):

    @classmethod
    def get_registered_class_type(cls):
        return SplitterDataset


class SplitterFactory(BaseFactory):

    @classmethod
    def get_registered_class_type(cls):
        return Splitter
