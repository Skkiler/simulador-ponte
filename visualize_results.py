import csv
from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.services.postprocessor import PostProcessor
from src.services.visualization_service import VisualizationService


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

if __name__ == "__main__":
    cfg = ConfigService().load()
    nodes, members, supports, loads = GeometryService().generate(cfg)
    node_results = read_csv("outputs/opensees/opensees_truss_nodes.csv")
    member_results = read_csv("outputs/opensees/opensees_truss_members.csv")
    member_checks = read_csv("outputs/opensees/member_failure_checks.csv")
    support_checks = read_csv("outputs/opensees/support_reaction_checks.csv")
    active = {int(r["node_id"]): str(r.get("support_active_vertical", "False")).lower() == "true" for r in node_results}
    supports2 = [type(s)(s.node_id, s.UX, s.UY, s.UZ if active.get(s.node_id, False) else 0, s.RX, s.RY, s.RZ, s.support_group, active.get(s.node_id, False)) for s in supports]
    paths = VisualizationService().save_all(nodes, members, supports2, loads, node_results, member_results, member_checks, support_checks, "outputs/plots", cfg["analysis"].get("deformed_scale", 30.0))
    for p in paths:
        print(f"Imagem/visual salvo: {p}")
