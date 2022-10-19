import sys
import os

from serotiny.ml_ops import _do_model_op_wrapper, _do_model_op

import logging
import warnings
import sys

from omegaconf import OmegaConf
from hydra.utils import get_original_cwd

def print_help():
    from textwrap import dedent

    print(
        dedent(
            """
    Usage:
      serotiny COMMAND

    Valid COMMAND values:
      train - train a model
      test - test a model
      predict - use a trained model to output predictions
      config - create a config yaml, given a Python class/function
      dataframe - utils to manipulate .csv dataframes
      image - image operations

    For more info on each command do:")
      serotiny COMMAND --help")
    """
        ).strip()
    )


def main():
    if sys.argv[0].endswith("run.py"):
        try:
            mode = sys.argv.pop(1)
        except IndexError:
            mode = "help"
    elif sys.argv[0].endswith("run.py.train"):
        mode = "train"
    elif sys.argv[0].endswith("run.py.test"):
        mode = "test"
    elif sys.argv[0].endswith("run.py.predict"):
        mode = "predict"
    else:
        raise NotImplementedError(f"Unknown command: '{sys.argv[0]}")

    if "help" in mode or mode == "-h":
        print_help()
        return

    # hydra modes
    if mode in ["train", "test", "predict"]:
        import hydra

        if sys.argv[0].endswith("run.py"):
            sys.argv[0] += f".{mode}"

        os.environ["HYDRA_FULL_ERROR"] = "1"

        def _do_model_op_wrapper(cfg):
            if isinstance(cfg, dict):
                cfg["mode"] = "test"
                cfg = OmegaConf.create(cfg)

            # there might be other dots in the
            # executable path, ours is the last
            mode = "test"
            if mode in ["train", "predict", "test"]:
                cfg = OmegaConf.merge(cfg, {"mode": mode})

            _do_model_op(
                **cfg,
                full_conf=cfg,
            )

        hydra.main(config_path=None, config_name=mode, version_base=None)(
            _do_model_op_wrapper
        )()

    # fire modes
    else:
        from fire import Fire

        import serotiny.cli.config_cli as config_cli
        import serotiny.cli.image_cli as image_cli
        from serotiny.cli.dataframe_cli import DataframeTransformCLI as dataframe_cli

        sys.argv.insert(1, mode)
        cli_dict = {
            "config": config_cli,
            "image": image_cli,
            "dataframe": dataframe_cli,
        }

        if mode in cli_dict:
            Fire(cli_dict)
        else:
            print_help()


if __name__ == "__main__":
    main()