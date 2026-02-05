import torch
from typing import List, Tuple, Dict

# Class added because torch.nn.ModuleDict doesn't allow certain keys to be used if they conflict with existing class attributes
# https://github.com/pytorch/pytorch/issues/71203
SUFFIX = "____"
SUFFIX_LENGTH = len(SUFFIX)
class RobustModuleDict(torch.nn.ModuleDict):
    """
    Torch ModuleDict wrapper that permits keys with any name, including those
    that would otherwise conflict with class attributes.

    Torch's `ModuleDict` does not allow certain keys (e.g., 'type', 'forward') 
    because they clash with existing methods or attributes of `nn.Module`, raising 
    errors like `KeyError`.

    Example:
        > torch.nn.ModuleDict({'type': torch.nn.Module()})  # Raises KeyError.
        > RobustModuleDict({'type': torch.nn.Module()})  # Works fine.

    This class mitigates possible conflicts by using a key-suffixing protocol.

    Args:
        init_dict (Dict[str, torch.nn.Module], optional): Initial dictionary of modules.
            If provided, it initializes the `RobustModuleDict` with these modules.
            Defaults to None.
    Returns:
        None
    """

    def __init__(self, init_dict: Dict[str, torch.nn.Module] = None) -> None:
        super().__init__()
        #self.module_dict = torch.nn.ModuleDict()
        if init_dict is not None:
            self.update(init_dict)

    # Retrieve a module using the original (unsuffixed) key.
    # Internally appends the suffix to avoid naming conflicts.
    def __getitem__(self, key) -> torch.nn.Module:
        return super().__getitem__(key + SUFFIX)

    # Set a module using the original (unsuffixed) key.
    # Internally appends the suffix to avoid naming conflicts.
    def __setitem__(self, key: str, module: torch.nn.Module) -> None:
        super().__setitem__(key + SUFFIX, module)

    # Return the number of modules in the dictionary.
    def __len__(self) -> int:
        return super().__len__()

    # Return a list of keys without the suffix.
    # This allows users to access the original keys.
    def keys(self) -> List[str]:
        return [key[:-SUFFIX_LENGTH] for key in super().keys()]

    # Return a list of modules.
    def values(self) -> List[torch.nn.Module]:
        return list(super().values())

    # def values(self) -> List[torch.nn.Module]:
    #     return [module for _, module in self.items()]

    # Return a list of (key, module) tuples without the suffix in the keys.
    # This allows users to access both the original keys and their corresponding modules.
    def items(self) -> List[Tuple[str, torch.nn.Module]]:
        return [
            (key[:-SUFFIX_LENGTH], module)
            for key, module in super().items()
        ]

    # Update the dictionary with a new set of modules.
    def update(self, modules: Dict[str, torch.nn.Module]) -> None:
        for key, module in modules.items():
            self[key] = module

    # def __next__(self) -> None:
    #     return next(iter(self))

    # Allow iteration over the keys of the dictionary.
    def __iter__(self):
        return iter(self.keys())
    
    # Check if a key exists in the dictionary.
    def __contains__(self, key: str) -> bool:
        return super().__contains__(key + SUFFIX)
    