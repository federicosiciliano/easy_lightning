import torch
from typing import List, Tuple, Dict

# Class added because torch.nn.ModuleDict doesn't allow certain keys to be used if they conflict with existing class attributes
# https://github.com/pytorch/pytorch/issues/71203
SUFFIX = "____"
SUFFIX_LENGTH = len(SUFFIX)
class RobustModuleDict(torch.nn.ModuleDict):
    """Torch ModuleDict wrapper that permits keys with any name.

    Torch's ModuleDict doesn't allow certain keys to be used if they
    conflict with existing class attributes, e.g.

    > torch.nn.ModuleDict({'type': torch.nn.Module()})  # Raises KeyError.

    This class is a simple wrapper around torch's ModuleDict that
    mitigates possible conflicts by using a key-suffixing protocol.
    """

    def __init__(self, init_dict: Dict[str, torch.nn.Module] = None) -> None:
        super().__init__()
        #self.module_dict = torch.nn.ModuleDict()
        if init_dict is not None:
            self.update(init_dict)

    def __getitem__(self, key) -> torch.nn.Module:
        return super().__getitem__(key + SUFFIX)

    def __setitem__(self, key: str, module: torch.nn.Module) -> None:
        super().__setitem__(key + SUFFIX, module)

    def __len__(self) -> int:
        return super().__len__()

    def keys(self) -> List[str]:
        return [key[:-SUFFIX_LENGTH] for key in super().keys()]

    def values(self) -> List[torch.nn.Module]:
        return list(super().values())

    # def values(self) -> List[torch.nn.Module]:
    #     return [module for _, module in self.items()]

    def items(self) -> List[Tuple[str, torch.nn.Module]]:
        return [
            (key[:-SUFFIX_LENGTH], module)
            for key, module in super().items()
        ]

    def update(self, modules: Dict[str, torch.nn.Module]) -> None:
        for key, module in modules.items():
            self[key] = module

    # def __next__(self) -> None:
    #     return next(iter(self))

    def __iter__(self):
        return iter(self.keys())
    
    def __contains__(self, key: str) -> bool:
        return super().__contains__(key + SUFFIX)
    