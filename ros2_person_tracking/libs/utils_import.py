"""
Utility module to handle dynamic imports.
This module allows to search for datasets, models, etc., in this repo and in external modules.
Classes from external modules may be overlayered by adding a class with the same name in this repo without requiring to change names in configs or in imports throughout the codebase.
"""

import inspect


def import_cost(name):
    import ros2_person_tracking.costs as costs

    modules = [costs]

    for module in modules:
        if hasattr(module, name):
            class_or_function_found = getattr(module, name)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                return class_or_function_found

    raise ImportError(f"Class or factory function {name} not found")
