from src.services.config_service import ConfigService
from src.services.pipeline import SimulationPipeline

if __name__ == "__main__":
    cfg = ConfigService().load()
    result = SimulationPipeline("outputs").run(cfg)
    print("Simulação concluída.")
    print("Status solver:", result["metrics"]["solver_status"])
    print("Frame3DD:", result["metrics"]["frame3dd_status"])
    print("Menor FS membros principais:", result["metrics"]["min_fs_primary"])
    print("Relatório:", result["report_path"])
    print("ZIP de resultados:", result["zip_path"])
