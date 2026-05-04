# Ponte de Palitos — Simulador Local v4

App local para projetar pontes de palitos com geometria, treliça, massa, palitos, colagens, falhas prováveis, lista de cortes, visualizações, Frame3DD e comparação de propostas.

## Rodar

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run app.py
```

## Novidades v4

- Correção do Frame3DD: densidade positiva para evitar `mass density d is not positive`.
- Entrada explícita do comprimento real do palito.
- UI mais simples e orientada a decisões.
- Comparação automática entre Parker, Pratt, Howe e Warren.
- Perfis superiores alternativos: `parker_plateau`, `triangular_peak`, `shallow_arch`, `flat`.
- Aba para localizar visualmente membros críticos.
- Gabaritos, cortes e palito por palito em `outputs/details` e `outputs/plots`.
- Proposta recomendada em `outputs/optimization/recommended_config.json`.

## Frame3DD

O app gera `outputs/frame3dd/ponte_palitos.3dd` e tenta rodar o executável local. A densidade positiva não adiciona peso próprio porque a gravidade do caso estático está zerada.

## Limitações

Modelo rápido para projeto amador: treliça axial, flambagem por Euler e estimativa discreta de palitos/cola. Não substitui ensaio físico nem FEM não linear.
