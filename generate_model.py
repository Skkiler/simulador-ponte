from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.solvers.frame3dd_adapter import Frame3DDAdapter

if __name__ == "__main__":
    cfg = ConfigService().load()
    geometry = GeometryService()
    nodes, members, supports, loads = geometry.generate(cfg)
    geometry.export_csvs(cfg, "outputs/model")
    path = Frame3DDAdapter().write_input(cfg, nodes, members, supports, loads, "outputs/frame3dd/ponte_palitos.3dd")
    print(f"Nós: {len(nodes)}")
    print(f"Membros: {len(members)}")
    print(f"Apoios modelados: {len(supports)}")
    print(f"Nós carregados: {len(loads)}")
    print(f"Frame3DD: {path}")
