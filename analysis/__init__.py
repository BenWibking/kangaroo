from .plotfile import PlotfileReader

__all__ = [
    "Runtime",
    "hpx_configuration_string",
    "Plan",
    "Stage",
    "TaskTemplate",
    "Domain",
    "FieldRef",
    "LoweringContext",
    "Dataset",
    "open_dataset",
    "RunMeta",
    "StepMeta",
    "LevelMeta",
    "LevelGeom",
    "BlockBox",
    "load_runmeta_from_dict",
    "PlotfileReader",
]


def __getattr__(name):
    if name in {"Runtime", "hpx_configuration_string"}:
        from .runtime import Runtime, hpx_configuration_string

        return Runtime if name == "Runtime" else hpx_configuration_string
    if name in {"Plan", "Stage", "TaskTemplate", "Domain", "FieldRef"}:
        from .plan import Plan, Stage, TaskTemplate, Domain, FieldRef

        return {
            "Plan": Plan,
            "Stage": Stage,
            "TaskTemplate": TaskTemplate,
            "Domain": Domain,
            "FieldRef": FieldRef,
        }[name]
    if name == "LoweringContext":
        from .ctx import LoweringContext

        return LoweringContext
    if name in {"Dataset", "open_dataset"}:
        from .dataset import Dataset, open_dataset

        return Dataset if name == "Dataset" else open_dataset
    if name in {
        "RunMeta",
        "StepMeta",
        "LevelMeta",
        "LevelGeom",
        "BlockBox",
        "load_runmeta_from_dict",
    }:
        from .runmeta import RunMeta, StepMeta, LevelMeta, LevelGeom, BlockBox, load_runmeta_from_dict

        return {
            "RunMeta": RunMeta,
            "StepMeta": StepMeta,
            "LevelMeta": LevelMeta,
            "LevelGeom": LevelGeom,
            "BlockBox": BlockBox,
            "load_runmeta_from_dict": load_runmeta_from_dict,
        }[name]
    raise AttributeError(name)
