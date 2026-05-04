# Planejador Ativo de Ponte de Palitos

Aplicação local para **planejar automaticamente** uma ponte treliçada (foco em ponte de palitos) a partir de **limites e metas**, em vez de exigir um desenho fechado pré-definido.

Você informa:
- limites geométricos (mín/máx de vão, largura, altura e painel);
- restrições de massa (máxima e alvo);
- propriedades mecânicas e geométricas dos palitos;
- propriedades da cola;
- carga de projeto, carga de ruptura alvo e FS mínimo.

O sistema então:
- gera propostas de configuração;
- filtra por etapas;
- refina automaticamente os candidatos mais promissores;
- cria versões finais construtivas com arredondamento conservador;
- executa análise estrutural e detalhamento;
- entrega o melhor candidato encontrado, alternativas e relatórios.

## O que o projeto faz

- Gera geometria 3D de treliça com variações de tipologia (Parker/Pratt/Howe/Warren), perfis de topo e contraventamentos.
- Resolve esforços axiais/reações/deslocamentos com solver matricial linear 3D.
- Faz checagem de capacidade (tração, compressão e flambagem por Euler).
- Estima detalhamento peça-a-peça (cortes, emendas, área de cola, massa, juntas críticas).
- Produz visualizações 3D/2D e arquivos CSV/JSON de engenharia.
- Compara candidatos de forma multiestágio e escolhe automaticamente o melhor pela função de score.

## Lógica de seleção de propostas

A busca é multiestágio:

1. `S1` varredura ampla
- Explora envelope geométrico + tipologias + perfis de reforço.
- Objetivo: cobertura alta de espaço de soluções.

2. `S2` refinamento local de palitos por grupo
- Parte dos melhores de `S1` e ajusta quantidades por grupos estruturais.
- Objetivo: melhorar resistência/massa sem explodir combinações.

3. `S3` validação detalhada
- Executa candidatos de `S2` com detalhamento peça-a-peça e cola.
- Objetivo: remover falsos positivos da análise simplificada.

4. `S4` refinamento adaptativo (iterativo)
- Parte dos melhores de `S3`.
- Detecta membros críticos e ajusta automaticamente:
  - quantidade de palitos por grupo crítico;
  - suporte pad e bracings quando necessário;
  - altura/painel em casos severos de flambagem.
- Reavalia em iterações até atingir limite de passos ou não haver melhorias.

5. `Final` trio de saída (`ideal`, `min`, `max`)
- `ideal`: melhor candidato matemático encontrado.
- `min`: arredondamento conservador para baixo dos comprimentos finais (ex.: 12,45 cm -> 12,0 cm).
- `max`: arredondamento conservador para cima (ex.: 12,45 cm -> 12,5 cm).
- Cada versão é recalculada e exportada com seus próprios FS, massa e ruptura estimada.

### Função objetivo (score)

Cada candidato recebe score por combinação ponderada de:
- aderência ao FS alvo;
- aderência à carga de ruptura alvo;
- aderência à massa alvo;
- margem para massa máxima.

Penalidades relevantes:
- FS abaixo do alvo;
- massa acima do limite;
- solver irregular;
- apoio crítico;
- erro de equilíbrio excessivo;
- perda de contato em apoios (uplift).

Perfis de objetivo disponíveis:
- `balanced` (equilíbrio geral);
- `max_strength` (prioriza segurança/capacidade);
- `min_mass` (prioriza leveza mantendo margem).

Além do perfil, a UI permite ajuste fino dos pesos de:
- FS;
- carga de ruptura;
- massa alvo;
- massa limite.

## O que ele retorna

No final de cada execução, o pipeline retorna e exporta:

- melhor configuração encontrada (`recommended_config.json`);
- configurações finais (`recommended_config_ideal.json`, `recommended_config_min.json`, `recommended_config_max.json`);
- ranking e rastreio das etapas (`active_stage1.csv`, `active_stage2.csv`, `active_stage3.csv`, `active_stage4_trace.csv`, `active_stage4.csv`);
- comparativo final (`active_final_variants.csv`, `recommended_final_variants_summary.json`);
- modelo estrutural analisado (`outputs/model/*.csv`);
- resultados do solver (`outputs/opensees/*.csv`);
- detalhamento de montagem/cola (`outputs/details/*.csv`);
- visualizações (`outputs/plots/*`);
- relatório consolidado (`outputs/reports/relatorio_automatico.md`);
- pacote único (`outputs/resultados_simulacao.zip`).

## Como usar

## 1) Instalação

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 2) Rodar interface

```powershell
streamlit run app.py
```

## 3) Fluxo recomendado na UI

1. Preencha objetivos e restrições.
2. Defina envelope geométrico mínimo/máximo.
3. Informe propriedades reais dos palitos e cola.
4. Escolha estratégia de objetivo (`balanced`, `max_strength`, `min_mass`).
5. (Opcional) Ajuste pesos da função objetivo no painel avançado.
6. Ajuste tamanho da busca (quantidade de propostas e iterações adaptativas).
7. Execute e analise as abas:
- etapas de busca;
- membros críticos;
- visual 3D/2D;
- montagem e cola;
- relatório/exports.

## 4) Rodar por CLI

```powershell
./venv/Scripts/python.exe run_cli.py
```

## Estrutura de código (visão geral)

- `app.py`: UI Streamlit (entrada por limites + análise dos resultados).
- `src/services/active_design_planner.py`: motor de busca multiestágio e seleção.
- `src/services/pipeline.py`: orquestra planejamento + análise + relatórios.
- `src/services/geometry_service.py`: geração geométrica e cargas/apoios.
- `src/solvers/linear_truss_solver.py`: solver axial linear 3D.
- `src/services/postprocessor.py`: checagens de capacidade e riscos.
- `src/services/stick_detail_service.py`: detalhamento palito/cola/cortes/massa.
- `src/services/visualization_service.py`: plots 3D/2D interativos e estáticos.

## Limitações importantes

- Modelo estrutural atual é linear-axial (não é FEM não linear completo).
- Flambagem é estimada por Euler (aproximação).
- Cola e emendas são estimativas de engenharia preliminar.
- Resultado deve ser validado com protótipo físico/ensaio real.

## Próximos ajustes sugeridos

- Calibrar propriedades mecânicas com ensaios do seu lote real de palitos e cola.
- Ajustar pesos da função objetivo para refletir a estratégia da competição.
- Incluir critérios adicionais de construtibilidade (tempo de montagem, simetria, tolerâncias).
