from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

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
    "Pipeline",
    "FieldHandle",
    "Histogram1DHandle",
    "Histogram2DHandle",
    "ParticleArrayHandle",
    "ParticleMaskHandle",
    "pipeline",
    "cdf_from_histogram",
    "cdf_from_samples",
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
    if name == "PlotfileReader":
        from .plotfile import PlotfileReader

        return PlotfileReader
    if name in {
        "Pipeline",
        "FieldHandle",
        "Histogram1DHandle",
        "Histogram2DHandle",
        "ParticleArrayHandle",
        "ParticleMaskHandle",
        "pipeline",
    }:
        from .pipeline import (
            FieldHandle,
            Histogram1DHandle,
            Histogram2DHandle,
            ParticleArrayHandle,
            ParticleMaskHandle,
            Pipeline,
            pipeline,
        )

        return {
            "Pipeline": Pipeline,
            "FieldHandle": FieldHandle,
            "Histogram1DHandle": Histogram1DHandle,
            "Histogram2DHandle": Histogram2DHandle,
            "ParticleArrayHandle": ParticleArrayHandle,
            "ParticleMaskHandle": ParticleMaskHandle,
            "pipeline": pipeline,
        }[name]
    if name in {"cdf_from_histogram", "cdf_from_samples"}:
        from .ops import cdf_from_histogram, cdf_from_samples

        return {
            "cdf_from_histogram": cdf_from_histogram,
            "cdf_from_samples": cdf_from_samples,
        }[name]
    raise AttributeError(name)
