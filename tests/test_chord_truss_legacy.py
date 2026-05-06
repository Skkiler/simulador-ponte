from __future__ import annotations

from copy import deepcopy

from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService


def test_chord_truss_legacy_migrates_with_warning(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["bridge"]["chord_truss_type"] = "Warren"
    cfg["bridge"].pop("top_chord_truss_type", None)
    cfg["bridge"].pop("bottom_chord_truss_type", None)
    cfg["bridge"]["legacy_chord_truss_lacing_enabled"] = False

    normalized = ConfigService().normalize(cfg)
    assert normalized["bridge"]["top_chord_truss_type"] == "Warren"
    assert normalized["bridge"]["bottom_chord_truss_type"] == "Warren"
    warnings = normalized.get("compatibility_warnings", [])
    assert any("chord_truss_type legado" in w for w in warnings)


def test_chord_truss_legacy_lacing_requires_explicit_flag(base_cfg: dict) -> None:
    cfg = base_cfg
    cfg["bridge"]["chord_truss_type"] = "X"
    cfg["bridge"]["legacy_chord_truss_lacing_enabled"] = False
    normalized = ConfigService().normalize(cfg)
    _, members_off, _, _ = GeometryService().generate(normalized)
    assert not any(m.group == "chord_lacing" for m in members_off)

    cfg_on = deepcopy(base_cfg)
    cfg_on["bridge"]["chord_truss_type"] = "X"
    cfg_on["bridge"]["legacy_chord_truss_lacing_enabled"] = True
    normalized_on = ConfigService().normalize(cfg_on)
    _, members_on, _, _ = GeometryService().generate(normalized_on)
    assert any(m.group == "chord_lacing" for m in members_on)
