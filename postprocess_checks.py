import csv
from pathlib import Path

from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.services.postprocessor import PostProcessor


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

if __name__ == "__main__":
    cfg = ConfigService().load()
    nodes, members, supports, loads = GeometryService().generate(cfg)
    member_results = read_csv("outputs/opensees/opensees_truss_members.csv")
    node_results = read_csv("outputs/opensees/opensees_truss_nodes.csv")
    # Apoios ativos vindos do resultado nodal
    active = {int(r["node_id"]): str(r.get("support_active_vertical", "False")).lower() == "true" for r in node_results}
    supports2 = [type(s)(s.node_id, s.UX, s.UY, s.UZ if active.get(s.node_id, False) else 0, s.RX, s.RY, s.RZ, s.support_group, active.get(s.node_id, False)) for s in supports]
    post = PostProcessor()
    member_checks = post.check_members(cfg, member_results)
    support_checks = post.check_supports(cfg, nodes, supports2, node_results)
    post.export(member_checks, support_checks, "outputs/opensees")
    print("Checagem de membros salva em: outputs/opensees/member_failure_checks.csv")
    print("Checagem de apoios salva em: outputs/opensees/support_reaction_checks.csv")
