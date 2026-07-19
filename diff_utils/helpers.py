"""Small workflow helpers retained by the training entry point."""

import os


def save_code_to_conf(conf_dir):
    path = os.path.join(conf_dir, "code")
    os.makedirs(path, exist_ok=True)
    for folder in ["utils", "models", "diff_utils", "dataloader", "metrics"]:
        target = os.path.join(path, folder)
        os.makedirs(target, exist_ok=True)
        os.system(f'cp -r ./{folder}/* "{target}"')
    os.system(f'cp *.py "{path}"')

