from src.services.config_service import ConfigService
from src.services.geometry_service import GeometryService
from src.solvers.linear_truss_solver import LinearTrussSolver

if __name__ == "__main__":
    cfg = ConfigService().load()
    nodes, members, supports, loads = GeometryService().generate(cfg)
    result = LinearTrussSolver().solve(nodes, members, supports, loads, unilateral_supports=cfg["bridge"].get("unilateral_supports", True))
    LinearTrussSolver().export(result, "outputs/opensees")
    print(f"Solver NumPy de treliça concluído. status={result.status}")
    print(f"Iterações contato unilateral: {result.iterations}")
    print(f"Erro equilíbrio vertical: {result.equilibrium_error_N:.3e} N")
    print("Arquivos gerados:")
    print(" - outputs/opensees/opensees_truss_nodes.csv")
    print(" - outputs/opensees/opensees_truss_members.csv")
