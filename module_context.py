import importlib
import sys

class ModuleContext:
    def __init__(self, *modules):
        self.modules = modules
        self.loaded_modules = {}

    def __enter__(self):
        for module_name in self.modules:
            self.loaded_modules[module_name] = importlib.import_module(module_name)
        return self.loaded_modules

    def __exit__(self, exc_type, exc_value, traceback):
        for module_name, module in self.loaded_modules.items():
            if module_name in sys.modules:
                del sys.modules[module_name]
            del module
        self.loaded_modules = {}

