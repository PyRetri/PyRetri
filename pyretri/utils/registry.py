# -*- coding: utf-8 -*-


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    A helper class to register class.
    """
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module):
        _register_generic(self, module.__name__, module)
        return module
