from .detector import FlashOCCModel


def build_flashocc_model(cfg, test_cfg=None):
    cfg = dict(cfg)
    cfg.pop("train_cfg", None)
    cfg.pop("test_cfg", None)
    model_type = cfg.pop("type")
    if model_type != "BEVDetOCC":
        raise KeyError(f"Unsupported standalone FlashOCC detector type: {model_type}")
    return FlashOCCModel(**cfg)
