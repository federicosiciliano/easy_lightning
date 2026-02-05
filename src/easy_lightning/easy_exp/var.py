# YAML parsing
yaml_additional_char = "+"  # Character indicating additional configurations
yaml_reference_char = "$"   # Character indicating references to other keys
yaml_raise_char = "^"       # Character indicating raised keys
yaml_nosave_char = "/"      # Character indicating keys marked as nosave
yaml_argparse_char = "-"    # Character indicating argparse argument keys
yaml_sweep_char = "£"      # Character indicating keys marked for sweep


yaml_global_key = "__global__"    # Key for global variables in YAML
yaml_imports_key = "__imports__"  # Key for imports in YAML
yaml_skip_key = "__"             # Key used to skip parsing a section in YAML

experiment_universal_key = "__exp__"          # Key for universal experiment configuration
experiment_nosave_key = "__nosave__"          # Key indicating nosave keys in an experiment
experiment_sweep_key = "__sweep__"          # Key indicating sweep keys in an experiment
