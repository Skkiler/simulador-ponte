# Planejador Ativo de Ponte de Palitos

Documentação técnico-científica do sistema de automação para geração, seleção, análise estrutural, refinamento e detalhamento construtivo de pontes treliçadas de palitos. O projeto implementa um pipeline computacional que parte de restrições geométricas, propriedades mecânicas, metas de ruptura e limite de massa, gera alternativas estruturais, avalia essas alternativas por modelos progressivamente mais rigorosos e exporta relatórios, listas de corte, planos de emenda, visualizações e pacotes de fabricação.

A aplicação foi projetada para uso local, com execução por interface Streamlit ou por linha de comando. O modelo físico é voltado a pontes treliçadas construídas com palitos de madeira e cola, com ênfase em resistência específica, massa competitiva, simetria estrutural, estabilidade global e viabilidade de montagem.

> Aviso técnico: o sistema fornece uma análise estrutural computacional preliminar. Ele não substitui ensaio físico, validação experimental, controle de qualidade do material, verificação normativa ou responsabilidade técnica profissional. Palitos, cola, umidade, defeitos locais, desalinhamento de montagem e cura podem alterar significativamente a capacidade real.

---

## Sumário

1. [Navegação rápida](#navegação-rápida)
2. [Fast try: execução mínima](#fast-try-execução-mínima)
3. [O que a automação faz](#o-que-a-automação-faz)
4. [Entradas, saídas e unidade física adotada](#entradas-saídas-e-unidade-física-adotada)
5. [Arquitetura geral do projeto](#arquitetura-geral-do-projeto)
6. [Lógica macro do pipeline](#lógica-macro-do-pipeline)
7. [Funil de fidelidade S0–S8](#funil-de-fidelidade-s0s8)
8. [Modelo geométrico da ponte](#modelo-geométrico-da-ponte)
9. [Modelo de cargas e distribuição de carregamentos](#modelo-de-cargas-e-distribuição-de-carregamentos)
10. [Solver estrutural matricial](#solver-estrutural-matricial)
11. [Pós-processamento estrutural](#pós-processamento-estrutural)
12. [Funções matemáticas e critérios de cálculo](#funções-matemáticas-e-critérios-de-cálculo)
13. [Módulos de decisão, reprovação e aceitação](#módulos-de-decisão-reprovação-e-aceitação)
14. [Detalhamento construtivo: palitos, emendas, cola e massa](#detalhamento-construtivo-palitos-emendas-cola-e-massa)
15. [Modelo de simetria e quarter-model](#modelo-de-simetria-e-quarter-model)
16. [Configurações do `bridge_config.json`](#configurações-do-bridge_configjson)
17. [Estrutura de diretórios e artefatos gerados](#estrutura-de-diretórios-e-artefatos-gerados)
18. [Execução pela interface Streamlit](#execução-pela-interface-streamlit)
19. [Execução por CLI](#execução-por-cli)
20. [Testes automatizados](#testes-automatizados)
21. [Limitações científicas e hipóteses de modelagem](#limitações-científicas-e-hipóteses-de-modelagem)
22. [Checklist operacional recomendado](#checklist-operacional-recomendado)

---

## Navegação rápida

Para testar imediatamente, use a seção [Fast try](#fast-try-execução-mínima). Para entender o funcionamento geral da automação, leia [Lógica macro do pipeline](#lógica-macro-do-pipeline) e depois [Funil de fidelidade S0–S8](#funil-de-fidelidade-s0s8). Para alterar comportamento, pesos, metas e restrições, vá direto para [Configurações do `bridge_config.json`](#configurações-do-bridge_configjson). Para interpretar resultados finais, leia [Estrutura de diretórios e artefatos gerados](#estrutura-de-diretórios-e-artefatos-gerados) e [Detalhamento construtivo](#detalhamento-construtivo-palitos-emendas-cola-e-massa).

---

## Fast try: execução mínima

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python run_cli.py
```

Para abrir a interface gráfica local:

```powershell
streamlit run app.py
```

Para rodar a suíte de testes:

```powershell
pytest -q
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python run_cli.py
```

Interface local:

```bash
streamlit run app.py
```

Testes:

```bash
pytest -q
```

Após a execução por CLI, o terminal imprime o status do solver interno, o status da validação Frame3DD, o menor fator de segurança dos membros principais, o caminho do relatório automático e o caminho do pacote `.zip` com resultados.

---

## O que a automação faz

O sistema automatiza o ciclo completo de projeto computacional de uma ponte treliçada de palitos. Ele não exige que o usuário forneça uma geometria final fechada. Em vez disso, recebe limites e metas de projeto, gera famílias de geometrias candidatas, distribui carregamentos, resolve esforços, calcula fatores de segurança, estima ruptura, refina dimensões e quantidade de palitos, detalha cortes e emendas, calcula massa instalada e massa de fabricação, valida regras de competição e exporta relatórios técnicos.

A automação atua em cinco frentes principais. Primeiro, ela gera o modelo estrutural tridimensional da ponte, incluindo nós, membros, apoios, contraventamentos e cargas. Segundo, resolve a estrutura por um solver matricial de treliça axial 3D, com opção de apoios unilaterais e membros tracionados apenas em casos configurados. Terceiro, avalia a capacidade estrutural por critérios de tração, compressão, flambagem, interação axial-flexão simplificada, apoio e cola. Quarto, usa um planejador ativo multiestágio para selecionar e refinar candidatos de forma progressiva. Quinto, transforma o modelo estrutural em documentação construtiva: peças, cortes, sobreposições, juntas, massa, agrupamento de montagem e pacote final.

O objetivo final não é apenas encontrar uma ponte resistente. O objetivo computacional é encontrar uma solução que maximize capacidade estrutural útil respeitando restrições de massa, geometria, simetria, construtibilidade, comprimento máximo de palitos, emendas, cola e estabilidade.

---

## Entradas, saídas e unidade física adotada

O arquivo de configuração principal é `bridge_config.json`. As unidades adotadas no projeto são `N-mm-MPa`. Isso significa que forças internas são tratadas em Newtons, comprimentos em milímetros e tensões em MPa, sendo `1 MPa = 1 N/mm²`. Cargas informadas em kgf são convertidas internamente usando:

```text
F_N = F_kgf · 9,80665
```

As entradas principais são: geometria admissível da ponte, vão, balanços laterais, número ou tamanho de painéis, largura, altura, perfil superior, tipo de treliça, modelo de carga, posição e área da placa de carga, propriedades do palito, propriedades da cola, resistência experimental por palito, limite de massa, fator de segurança mínimo, carga de projeto, carga de ruptura alvo e opções do planejador.

As saídas principais são: configuração usada, modelo estrutural em tabelas, resultados do solver, checagens por membro, checagens por apoio, estimativa de ruptura, rankings dos candidatos, rastros de decisão, listas de corte, plano de emendas, relatório automático, visualizações e pacotes `.zip`.

---

## Arquitetura geral do projeto

A raiz do repositório concentra a interface, configuração e entrada CLI:

```text
app.py                    Interface Streamlit.
run_cli.py                Execução direta do pipeline por terminal.
bridge_config.json        Configuração principal.
requirements.txt          Dependências Python.
tests/                    Testes automatizados.
Frame3DD/                 Binários e arquivos auxiliares do Frame3DD.
src/                      Código-fonte da automação.
outputs/                  Resultados gerados pela execução.
```

Os módulos centrais em `src/` são:

```text
src/services/config_service.py              Carregamento, defaults e normalização da configuração.
src/services/pipeline.py                    Orquestração geral da simulação.
src/services/staged_fidelity_funnel.py      Otimizador multiestágio S0–S8.
src/services/active_design_planner.py       Planejador ativo legado/compatível e sizing local.
src/services/geometry_service.py            Geração de nós, barras, apoios e cargas.
src/services/load_distribution_service.py   Distribuição física de carregamentos.
src/solvers/linear_truss_solver.py          Solver matricial linear de treliça 3D.
src/solvers/frame3dd_adapter.py             Exportação e execução opcional via Frame3DD.
src/services/postprocessor.py               Cálculo de FS, flambagem, apoio e métricas.
src/services/section_service.py             Propriedades geométricas e capacidade das seções.
src/services/connection_planner.py          Escolha de emendas por membro.
src/services/stick_detail_service.py        Lista de peças, cortes, massa e cola.
src/services/quarter_model_service.py       Simetria e modelo de um quarto da ponte.
src/services/mass_guard.py                  Semântica única do limite de massa.
src/services/rupture_estimator.py           Estimativa de carga de ruptura.
src/services/report_bundle_service.py       Relatórios finais e pacote de exportação.
src/services/visualization_service.py       Visualizações 2D/3D.
```

---

## Lógica macro do pipeline

O fluxo macro é coordenado por `SimulationPipeline`, em `src/services/pipeline.py`. A lógica pode ser descrita como uma cadeia de transformação e decisão:

```text
Configuração bruta
  ↓
Normalização e aplicação de defaults
  ↓
Planejamento ativo / otimização multiestágio
  ↓
Seleção de candidato viável ou fallback técnico
  ↓
Geração geométrica 3D
  ↓
Distribuição de cargas e apoios
  ↓
Solução estrutural matricial
  ↓
Pós-processamento de FS, ruptura e apoios
  ↓
Sizing local e eventual reanálise
  ↓
Validação externa opcional com Frame3DD
  ↓
Planejamento de emendas
  ↓
Detalhamento de palitos, cortes, cola e massa
  ↓
Geração de relatórios, visualizações e pacotes
```

A automação tenta resolver o problema em camadas. Primeiro procura uma topologia e geometria promissora. Depois avalia com casos de carga múltiplos. Depois faz refinamento geométrico. Depois ajusta quantidades de palitos por membro ou por grupo. Depois aplica regras de fabricação. Por fim, valida a solução contra critérios de massa, ruptura, fator de segurança, equilíbrio e construtibilidade.

Essa estratégia evita uma busca exaustiva cega. A quantidade de combinações possíveis de treliça, altura, painel, largura, seção, emenda e reforço é grande demais para avaliação detalhada integral. Por isso o projeto usa um funil de fidelidade: muitos candidatos são avaliados com critérios rápidos; poucos chegam às etapas caras de simulação, detalhamento e validação final.

---

## Funil de fidelidade S0–S8

O otimizador moderno está implementado principalmente em `src/services/staged_fidelity_funnel.py`. Ele organiza a busca em estágios numerados. Cada estágio aumenta a fidelidade do modelo ou refina candidatos já promissores.

### S0 — Validação do domínio

O estágio S0 verifica se a configuração está dentro do domínio físico e competitivo previsto. São avaliados vão, balanços, largura, altura, dimensões dos palitos, carga positiva, simetria e consistência das restrições. Candidatos que violam regras elementares são reprovados antes de consumir tempo computacional.

Critérios típicos avaliados nesta etapa incluem: vão configurado compatível com o domínio do projeto, balanços dentro do limite, largura entre limites admissíveis, altura central positiva e suficiente, dimensões de palitos válidas, carga de projeto positiva e coerência dos contatos de apoio. O estágio também registra diagnósticos de domínio, permitindo rastrear por que uma execução foi aceita ou recusada.

### S1 — Geração de arquétipos macro

O estágio S1 constrói famílias iniciais de pontes. O código trabalha com arquétipos como Pratt, Howe, Warren, Warren com verticais, variantes com contraventamento em X, reforço de apoio, variações de altura, ponte em caixa e perfis superiores tipo Parker ou platô. A função desse estágio é cobrir o espaço de soluções de forma ampla.

Cada arquétipo é convertido em uma configuração candidata. Essa configuração define geometria global, perfil do banzo superior, distribuição de painéis, largura, tipo de triangulação, quantidade inicial de palitos por grupo e reforços de estabilidade.

### S2 — Triagem rápida

O estágio S2 avalia os candidatos com fidelidade reduzida, normalmente usando caso de carga central e métricas rápidas de massa, equilíbrio e resistência. O objetivo é eliminar soluções inviáveis sem executar todo o detalhamento.

Reprovações comuns em S2: solver irregular, mecanismo estrutural, equilíbrio vertical ruim, massa muito acima do limite, ausência de caminho de carga, fator de segurança baixo demais, ruptura estimada insuficiente e geometria instável.

A etapa mantém apenas os melhores candidatos, respeitando diversidade para evitar que pequenas variações de uma mesma solução eliminem alternativas topologicamente diferentes.

### S3 — Triagem multi-loadcase

O estágio S3 executa casos de carga múltiplos. A ponte é avaliada sob carregamento central, carregamentos deslocados, torção por distribuição assimétrica, imperfeição lateral, contato em região de coroamento, placa isolada e peso próprio quando configurado.

A métrica de decisão passa a considerar o pior caso. A automação não seleciona apenas o candidato bom no carregamento ideal; ela procura candidatos que mantenham capacidade sob cenários assimétricos e menos favoráveis. São agregadas métricas como menor FS de projeto, menor ruptura estimada, deslocamento máximo, massa média, qualidade do caminho de carga e balanço de reações nos apoios.

### S4 — Refinamento geométrico local

O estágio S4 usa uma lógica de vizinhança local. Partindo de bons candidatos, testa variações de altura, largura e painel dentro de uma região de confiança. Se uma alteração melhora a função objetivo, a região pode ser mantida ou expandida. Se piora, os passos são reduzidos.

Essa etapa procura melhorar a ponte sem trocar completamente a família estrutural. Ela é especialmente relevante para compressão e flambagem, pois pequenas mudanças em altura, comprimento de painel e largura alteram o comprimento efetivo de membros comprimidos, a triangulação e o braço de alavanca estrutural.

### S5 — Dimensionamento de membros

O estágio S5 ajusta a quantidade de palitos por membro ou por grupos de membros. A decisão é baseada em utilização estrutural, fator de segurança, contribuição para massa e ganho esperado por grama adicionada. Membros críticos recebem reforço; membros com baixa utilização podem ser candidatos a redução, desde que isso não quebre simetria, estabilidade ou critérios mínimos de grupo.

O módulo registra trilhas de decisão como membros críticos, doadores de massa, reforços aplicados, antes/depois de dimensionamento e evolução de massa. A ideia é usar a massa disponível onde ela gera maior ganho estrutural marginal.

### S6 — Mutação topológica e ajustes tardios

O estágio S6 aplica limpezas e mutações estruturais. Ele pode remover membros de força baixa, ajustar padrões de painéis, reinvestir massa economizada, reforçar apoio, recuperar meta de ruptura sob casos assimétricos, aplicar trocas entre grupos, aparar massa respeitando simetria e executar auditorias finais.

Esse estágio é conservador. Remover massa de uma treliça pode criar mecanismos, piorar torção ou concentrar força em juntas. Por isso a limpeza topológica só é aceita quando a solução reavaliada preserva equilíbrio, solver regular, fatores de segurança mínimos e massa dentro do limite.

### S7 — Detalhamento de fabricação

O estágio S7 transforma a geometria estrutural em instruções construtivas. Ele calcula peças por membro, comprimentos de corte, sobreposições, emendas, massa de palitos instalados, massa de cola, desperdício, número de palitos comprados e relatórios de montagem.

Essa etapa é necessária porque uma ponte viável no modelo estrutural pode ser inviável no chão de montagem. Um membro de 420 mm, por exemplo, precisa ser fabricado com múltiplos palitos, emendas e sobreposições. Cada emenda muda massa, rigidez local, risco de falha e consumo de cola.

### S8 — Validação final

O estágio S8 executa a validação final da solução selecionada. O candidato é aprovado ou reprovado com base em solver, equilíbrio, massa de competição, ruptura prevista, fatores de segurança mínimos, apoio e critérios configurados.

A saída final contém o veredito e diagnósticos de casos de carga. Uma solução pode ser estruturalmente forte e ainda ser reprovada por massa. Também pode ser leve e ser reprovada por ruptura, apoio, flambagem, cola ou instabilidade numérica.

---

## Modelo geométrico da ponte

A geometria é gerada por `GeometryService`. O modelo é tridimensional e usa nós com coordenadas `(x, y, z)`. O eixo `x` representa o comprimento/vão, `y` representa a largura e `z` representa a altura. A ponte é formada por banzos inferiores, banzos superiores, diagonais, verticais, transversinas, contraventamentos e pads de apoio.

O perfil superior pode assumir diferentes funções. No perfil plano, a altura é constante. No perfil triangular, a altura cresce até o meio do vão e depois decresce. No perfil de arco raso, a altura segue uma parábola. No perfil Parker com platô, a altura cresce linearmente até uma região central, permanece constante e depois decresce.

Em forma genérica, para um perfil parabólico raso:

```text
h(x) = h_end + (h_center - h_end) · (1 - ξ²)
ξ = (x - span/2) / (span/2)
```

Para um perfil Parker com platô, a função é definida por trechos: rampa ascendente, platô central e rampa descendente. Isso permite aumentar altura onde o momento global tende a ser maior e reduzir altura perto dos apoios, preservando massa.

O comprimento de cada membro é calculado por distância euclidiana:

```text
L = √(Δx² + Δy² + Δz²)
```

Os cossenos diretores são:

```text
c = [Δx/L, Δy/L, Δz/L]
```

Esses valores são usados diretamente na matriz de rigidez axial do solver.

---

## Modelo de cargas e distribuição de carregamentos

A distribuição de carga é controlada por `LoadDistributionService` e pelas configurações de `bridge.load_model`, `bridge.load_footprint_*`, `multi_loadcase_screening` e campos relacionados. O sistema suporta carregamento pontual interpolado, carregamento distribuído por superfície de placa, carregamento com assimetria torsional, contato em região de coroamento, imperfeição lateral e peso próprio.

Quando a carga é aplicada por uma placa, o algoritmo identifica nós ou estações influenciadas pela área de contato. Os pesos são normalizados para que a soma das cargas aplicadas seja igual à carga total convertida para Newtons. Em casos torsionais, a distribuição entre lado esquerdo e direito é enviesada, por exemplo 60/40, 70/30 ou 80/20. Em casos de contato no coroamento, a carga se concentra em nós altos dentro da tolerância configurada. Em casos de imperfeição lateral, adiciona-se componente horizontal proporcional à carga vertical.

A lógica física é evitar uma hipótese excessivamente otimista. Uma ponte que suporta carga perfeitamente centralizada pode falhar quando a carga é deslocada, quando a placa toca mais um lado do que o outro ou quando há pequena componente lateral por montagem imperfeita.

---

## Solver estrutural matricial

O solver interno está em `src/solvers/linear_truss_solver.py`. Ele implementa um modelo de treliça axial linear em 3D. Cada nó possui três graus de liberdade translacionais. Cada membro transmite força axial ao longo de seu eixo. Não há, no solver básico, rigidez flexional completa de pórtico; a flexão entra posteriormente em checagens simplificadas de estabilidade e imperfeição.

A rigidez local axial do elemento é:

```text
k_local = (E · A / L) · c · cᵀ
```

onde `E` é o módulo de elasticidade, `A` é a área efetiva da seção, `L` é o comprimento do membro e `c` é o vetor de cossenos diretores. A contribuição do elemento na matriz global 6×6 dos dois nós é montada como:

```text
[ +k  -k ]
[ -k  +k ]
```

Depois da montagem global, os graus de liberdade restringidos por apoios são separados dos graus livres. O sistema reduzido é:

```text
K_ff · U_f = F_f
```

Se `K_ff` possui posto completo, a solução é obtida por solução direta. Se o sistema é singular ou quase singular, o solver retorna status irregular e pode usar solução por mínimos quadrados apenas como diagnóstico, não como aprovação estrutural robusta.

As reações são calculadas por:

```text
R = K · U - F
```

Os esforços axiais são obtidos pela projeção do deslocamento relativo dos nós na direção do membro. O sinal separa tração e compressão.

O solver também contempla lógica iterativa para apoios unilaterais e membros de tração apenas quando configurados. Em apoios unilaterais, uma reação vertical negativa além da tolerância indica perda de contato; o apoio pode ser removido da iteração. Em membros de tração apenas, barras comprimidas além da tolerância podem ser desativadas, desde que a estrutura resultante permaneça estável.

---

## Pós-processamento estrutural

O pós-processamento é feito por `PostProcessor`, `SectionService`, `MassGuard` e `RuptureEstimator`. A etapa recebe deslocamentos, reações e forças axiais, e produz métricas interpretáveis: fator de segurança por membro, fator de segurança dos apoios, menor FS de grupos primários, menor FS global, deslocamento máximo, ruptura estimada, massa de competição e status de aprovação.

Membros tracionados são avaliados principalmente por capacidade axial de tração. Membros comprimidos são avaliados por compressão direta, flambagem nos eixos principais e interação axial-flexão simplificada. Apoios são avaliados pela reação vertical e pela capacidade admissível de contato. Juntas de cola são avaliadas posteriormente no detalhamento, pois dependem do plano construtivo e da geometria da emenda.

A ruptura estimada usa o menor fator de segurança governante aplicado à carga de projeto:

```text
P_ruptura_estimada ≈ P_projeto · FS_governante
```

Esse valor é uma aproximação de engenharia. Ele assume proporcionalidade linear entre carregamento e esforços internos até o modo de falha governante, o que é coerente com a análise elástica linear, mas não captura redistribuição não linear, dano progressivo, plastificação da cola, fissuras, esmagamento local ou instabilidade pós-crítica.

---

## Funções matemáticas e critérios de cálculo

### Conversão de carga

```text
F_N = F_kgf · 9,80665
```

Essa conversão é usada para transformar carga de projeto, ruptura alvo e capacidades experimentais informadas em kgf para o sistema interno em Newtons.

### Área e inércia de seção retangular

Para um palito retangular idealizado:

```text
A = b · h
I_y = b · h³ / 12
I_z = h · b³ / 12
```

Para seções compostas por múltiplos palitos, o código calcula centroides e aplica o teorema dos eixos paralelos:

```text
I_total = Σ(I_i + A_i · d_i²)
```

A seção efetiva pode ser reduzida por fatores de eficiência `eta_A` e `eta_I`, representando ligação imperfeita, cola, espaçamento e comportamento não perfeitamente composto.

### Capacidade de tração

A capacidade de tração é tratada como proporcional ao número de palitos resistentes:

```text
T_allow = n_sticks · T_per_stick
```

O fator de segurança em tração é:

```text
FS_t = T_allow / N_tension
```

### Capacidade de compressão direta

A compressão usa ancoragens experimentais para 1 e 2 palitos e extrapolação limitada para mais palitos. O modelo evita assumir ganho linear indefinido, pois a compressão real sofre influência de instabilidade, excentricidade, colagem e imperfeições geométricas.

De forma simplificada, para múltiplos palitos:

```text
C_allow = min(capacidade_por_area, limite_multiplicador · capacidade_linear)
```

com piso definido por dados experimentais quando disponíveis. Esse mecanismo impede que adicionar palitos aumente artificialmente a capacidade sem limite físico.

### Flambagem de Euler

Para membros comprimidos, a carga crítica elástica é:

```text
P_cr = π² · E · I / (K · L)²
```

onde `K` é o fator de comprimento efetivo, `L` é o comprimento do membro e `I` é a inércia no eixo analisado. O fator de segurança à flambagem é:

```text
FS_buckling = P_cr / |N_compression|
```

Como a seção tem dois eixos principais, o sistema avalia flambagem em `y` e `z`, usando o menor resultado.

### Índice de esbeltez

O raio de giração é:

```text
r = √(I/A)
```

A esbeltez é:

```text
λ = K · L / r
```

Esse índice informa o quanto o membro comprimido é sensível à flambagem. Quanto maior `λ`, maior a tendência de falha por instabilidade antes do esmagamento direto.

### Transição Johnson/Euler

O modelo pode usar uma transição simplificada entre flambagem inelástica tipo Johnson e flambagem elástica de Euler. A fronteira típica usa:

```text
λ_lim = √(2π²E/σ_c)
```

Para membros menos esbeltos, a compressão pode ser limitada por tensão admissível corrigida. Para membros mais esbeltos, Euler tende a governar.

### Interação axial-flexão por imperfeição

A automação inclui uma aproximação de imperfeição geométrica. Um membro comprimido ou tracionado pode ter excentricidade residual `e`. O momento imperfeito é:

```text
M_imp = |N| · e_eff
```

A tensão de flexão aproximada é:

```text
σ_b = M · c / I
```

Em compressão, o efeito de segunda ordem é amplificado por um fator do tipo:

```text
B1 = 1 / (1 - |N|/P_cr)
```

com limites numéricos para evitar explosões instáveis. A utilização combinada considera parcela axial e parcela flexional. O fator de segurança governante é aproximadamente o inverso da utilização.

### Equilíbrio global

O equilíbrio vertical é avaliado por:

```text
erro_z = ΣF_z + ΣR_z
```

Um erro relativo pequeno indica que a soma das reações verticais compensa a carga aplicada. Erro elevado sugere problema de apoio, singularidade, perda de contato, má distribuição de carga ou falha de solver.

### Massa de competição

A massa de competição é tratada como massa efetivamente presente na ponte:

```text
m_competição = m_palitos_instalados + m_cola_curada
```

A massa de fabricação ou aquisição pode incluir desperdício, cortes, palitos comprados e cola úmida:

```text
m_fabricação = m_palitos_comprados + m_cola_úmida
```

O limite efetivo de massa é resolvido por `MassGuard`, combinando o limite do planejador e o limite material quando ambos existem. Isso evita que módulos diferentes usem limites divergentes.

### Score objetivo

A função objetivo do planejador combina resistência, segurança, massa e construtibilidade. Em termos conceituais:

```text
score = recompensa_resistência
      + recompensa_FS
      + recompensa_resistência_por_massa
      + recompensa_margem_de_massa
      + recompensa_construtibilidade
      - penalidade_massa_excedida
      - penalidade_mecanismo
      - penalidade_deslocamento
      - penalidade_cola
      - penalidade_complexidade
```

Perfis como `max_strength_per_competition_mass`, `max_strength`, `balanced` ou `min_mass` alteram a importância relativa das recompensas e penalidades.

---

## Módulos de decisão, reprovação e aceitação

A automação contém múltiplos pontos de decisão. Eles existem para impedir que uma solução aparentemente boa por uma métrica isolada seja aceita apesar de violar física, massa, fabricação ou estabilidade.

### Decisão por solver

Um candidato é penalizado ou reprovado se o solver retorna singularidade, posto insuficiente, mecanismo, perda de estabilidade após remoção de membros tension-only, perda excessiva de apoio unilateral ou erro de equilíbrio acima da tolerância. Um resultado numérico só é útil se a matriz estrutural representa uma estrutura estável.

### Decisão por massa

A massa é uma restrição dura quando `strict_mass_acceptance` está habilitado. Isso significa que uma ponte muito forte, mas acima do limite, não é aceita como solução final. O sistema pode tentar resgatar massa por redução de membros doadores, limpeza topológica ou ajustes de seção, mas a solução final precisa respeitar o limite efetivo.

### Decisão por fator de segurança

A automação separa FS global, FS de grupos primários, FS de apoio e FS de cola. Membros primários incluem elementos que participam diretamente do caminho principal de carga, como banzos, diagonais e verticais principais. Estabilizadores e contraventamentos também são importantes, mas podem ser tratados com limiares diferentes dependendo da configuração.

Uma ponte pode ser reprovada se o menor FS primário ficar abaixo do mínimo configurado, se apoio entrar em falha, se cola ficar abaixo do FS mínimo ou se a ruptura estimada não atingir a meta.

### Decisão por ruptura estimada

A ruptura estimada é calculada a partir do modo governante. Se o projeto exige 120 kgf e a ponte é analisada sob 80 kgf, um FS governante de 1,50 sugere ruptura estimada de aproximadamente 120 kgf. Se o FS governante for 1,20, a ruptura estimada seria aproximadamente 96 kgf e o candidato pode ser reprovado mesmo suportando a carga de projeto.

### Decisão por construtibilidade

O detalhamento pode reprovar ou penalizar soluções com emendas excessivamente fracas, comprimentos de corte inválidos, sobreposições insuficientes, consumo de cola incompatível, massa de fabricação exagerada, cluster de emendas alinhadas, falta de escalonamento ou complexidade topológica muito alta.

### Decisão por robustez multi-loadcase

Um candidato não é julgado apenas pelo carregamento central. Casos assimétricos e laterais podem governar a solução. Se uma ponte falha em torção 70/30 ou em carga deslocada, a automação evita aceitá-la como solução robusta, mesmo que ela pareça boa no caso central.

---

## Detalhamento construtivo: palitos, emendas, cola e massa

O detalhamento é executado por `StickDetailService` em conjunto com `ConnectionPlanner`, `SpliceStaggeringService`, `AssemblyGroupingService` e `AssemblyTutorialService`.

### Peças e cortes

Cada membro estrutural possui comprimento geométrico. Se o comprimento é menor ou igual ao comprimento de um palito, ele pode ser representado por uma peça única. Se excede o comprimento do palito, o sistema cria uma sequência de peças com sobreposição. A distância útil avançada por peça é aproximadamente:

```text
passo = comprimento_palito - sobreposição
```

O corte pode ser arredondado para incrementos construtivos, como 5 mm. O arredondamento conservador usa teto para evitar peça menor do que a necessária. Isso é importante porque erro para baixo pode reduzir sobreposição real e resistência da emenda.

### Planejamento de emendas

`ConnectionPlanner` escolhe o modelo de emenda por membro com base em estado de força, razão de força, FS, grupo estrutural e simetria. Membros com força quase nula podem receber emendas leves. Membros tracionados leves podem receber emenda simples. Membros comprimidos ou críticos podem receber emenda dupla ou reforçada.

A lógica usa a razão:

```text
force_ratio = |N_membro| / max(|N|)
```

Essa razão classifica a severidade da emenda. A severidade pode ser aumentada se o FS do membro está baixo, se o membro pertence a grupo primário ou se trabalha em compressão.

### Cola e cisalhamento

A área de cola é calculada em função da sobreposição, largura do palito e fator de área do tipo de junta:

```text
A_cola = sobreposição · largura_palito · fator_area_junta
```

A tensão média de cisalhamento na cola é:

```text
τ = demanda_por_faixa / A_cola
```

A tensão admissível usa o fator de segurança da cola:

```text
τ_adm = τ_resistência / FS_cola
```

O FS da cola é:

```text
FS_cola = τ_adm / τ_solicitada
```

Esse modelo é simplificado, mas evita ignorar uma falha comum em pontes de palito: a junta não falha pelo palito em si, mas por cisalhamento, destacamento ou má execução da cola.

### Massa de cola

A cola úmida é estimada por área colada e taxa de espalhamento:

```text
m_cola_úmida = A_cola_m² · taxa_espalhamento / eficiência
```

A massa curada é:

```text
m_cola_curada = m_cola_úmida · fração_sólidos
```

A competição normalmente considera a massa final da ponte, isto é, palitos instalados mais cola curada. O planejamento de compra e fabricação também pode considerar cola úmida e desperdício.

### Escalonamento de emendas

Emendas alinhadas em membros vizinhos podem gerar uma seção fraca transversal. O serviço de escalonamento tenta detectar e reduzir clusters de emendas. O objetivo é distribuir juntas ao longo do comprimento da ponte para evitar que várias descontinuidades fiquem na mesma estação `x`.

---

## Modelo de simetria e quarter-model

O projeto contém suporte a simetria estrutural e modelo de um quarto da ponte. Quando a geometria, cargas e apoios permitem, `QuarterModelService` pode reduzir o modelo usando planos de simetria em `x = span/2` e `y = 0`. A carga é ajustada por multiplicidade e os resultados podem ser replicados para reconstruir o comportamento global.

A simetria ajuda em duas frentes. Primeiro, reduz custo computacional em algumas avaliações. Segundo, força coerência construtiva: uma ponte simétrica tende a distribuir esforços de modo mais previsível e reduz erros de montagem. O sistema, porém, não assume simetria cegamente. Ele valida correspondência de nós, membros, apoios, cargas e propriedades. Se a simetria é inválida ou o quarter-model gera instabilidade, o pipeline retorna ao modelo completo.

Por padrão, a validação final pode ser refeita no modelo completo para evitar que uma simplificação simétrica esconda comportamento local assimétrico.

---

## Configurações do `bridge_config.json`

A configuração é carregada e normalizada por `ConfigService`. Defaults são aplicados para manter compatibilidade com versões anteriores. A seguir estão os blocos principais e o papel técnico de cada um.

### `units`

Define o sistema de unidades. O padrão do projeto é:

```json
"units": "N-mm-MPa"
```

Todos os cálculos estruturais esperam coerência com esse sistema.

### `project`

Contém metadados: nome do projeto, autor, versão e identificação geral. Esses dados aparecem em relatórios e pacotes de saída.

### `bridge`

Define geometria global e modelo de ponte. Campos relevantes incluem vão (`span_mm`), balanços laterais, tamanho do painel, largura, altura de extremidade, altura central, perfil superior, tipo de treliça, contraventamentos, pads de apoio, contatos de apoio, modelo de carga, carga em kgf e posição da placa de carga.

Exemplos de controles importantes:

```json
"span_mm": 1200,
"panel_mm": 100,
"width_mm": 160,
"end_height_mm": 100,
"center_height_mm": 300,
"truss_type": "pratt",
"top_profile": "parker_plateau",
"load_kgf": 80,
"load_model": "plate_surface_uniform"
```

A escolha de `panel_mm` influencia diretamente a esbeltez dos membros. A escolha de `center_height_mm` altera braço estrutural e compressão nos banzos. A largura afeta estabilidade torsional e contraventamentos.

### `material`

Define propriedades físicas dos palitos e cola. Inclui módulo de elasticidade, módulo de cisalhamento, dimensões do palito, massa por palito, resistência à tração por palito, resistência experimental à compressão, tensão de flexão, tensão de compressão, limite de massa e modelo de capacidade.

Campos críticos:

```json
"E_MPa": 6000,
"G_MPa": 500,
"stick_length_mm": 120,
"stick_width_mm": 7,
"stick_thickness_mm": 1.5,
"stick_mass_g": 1.4,
"tension_capacity_kgf_per_stick": 72,
"compression_capacity_kgf_one_stick": 4,
"compression_capacity_kgf_two_sticks": 11,
"mass_limit_g": 1000
```

Esses valores devem ser calibrados experimentalmente sempre que possível. Pequenas diferenças em palitos reais podem alterar fortemente flambagem e resistência de junta.

### `member_sticks_by_group`

Define quantidade inicial de palitos por grupo estrutural. Grupos típicos incluem banzo inferior, banzo superior, verticais, diagonais, transversinas, contraventamentos e pads de apoio.

A quantidade de palitos afeta área, inércia, massa, tração, compressão e flambagem. O planejador pode modificar essa distribuição se `member_sizing` estiver habilitado.

### `effective_length_factor_by_group`

Define fatores de comprimento efetivo `K` por grupo e eixo de flambagem. O fator `K` aparece em:

```text
P_cr = π²EI/(KL)²
```

Um `K` menor representa melhor restrição lateral ou condição de extremidade mais favorável. Um `K` maior torna a flambagem mais severa.

### `support_check`

Controla a verificação dos apoios. Inclui comprimento de contato, quantidade de palitos em contato, capacidade admissível por palito de contato, tratamento de uplift e critérios de reação.

Esse bloco é importante porque uma ponte pode não falhar no vão, mas esmagar ou perder contato nos apoios.

### `planner`

Define metas e limites do planejador: carga alvo, ruptura alvo, massa máxima, massa alvo, limiares de sizing, número de iterações locais, regras de reforço e mínimos por grupo.

Campos típicos:

```json
"target_load_kgf": 80,
"target_breaking_load_kgf": 120,
"max_bridge_mass_g": 1000,
"target_mass_g": 950
```

O planejador usa esses dados para decidir se deve reforçar, aparar massa ou reprovar o candidato.

### `analysis`

Define critérios estruturais e estratégia de análise. Inclui execução do Frame3DD, alvo de FS, mínimos de aceitação, ruptura de projeto, perfil objetivo, grupos primários, grupos estabilizadores, otimização de variantes, aceitação estrita de massa e simetria.

Exemplos:

```json
"target_min_fs": 1.5,
"acceptance_min_fs_primary": 1.05,
"acceptance_min_fs_support": 1.0,
"acceptance_min_fs_glue": 1.5,
"design_breaking_load_kgf": 120,
"objective_profile": "max_strength_per_competition_mass",
"optimize_variants": true,
"strict_mass_acceptance": true
```

### `detail_model`

Configura o modelo de fabricação: tipo de emenda, resistência da cola, taxa de espalhamento, eficiência de aplicação, fração de sólidos, FS de junta, desperdício, largura de corte, imperfeição geométrica, incremento de corte, comprimento máximo de corte e política para contraventamentos cruzados.

Esse bloco conecta o modelo estrutural ao modelo físico de montagem.

### `section_layout_by_group`

Define arranjos de seção por grupo. O mesmo número de palitos pode gerar propriedades diferentes dependendo da disposição: pilha simples, caixa, seção espaçada ou layout composto. Esse bloco influencia inércia, raio de giração, flambagem e massa.

### `multi_loadcase_screening`

Define casos de carga usados na triagem e validação. Casos comuns incluem carga central, deslocamentos laterais, torção 60/40, 70/30 e 80/20, imperfeição lateral, placa única, contato no coroamento e peso próprio.

A robustez da ponte depende fortemente desse bloco. Se poucos casos são ativados, a busca pode produzir uma solução otimizada demais para uma condição específica.

### `member_sizing`

Controla reforço e redução de membros. Inclui passes tardios, aparo simétrico, reinvestimento de massa, reforço de suporte, recuperação de casos assimétricos, topoff nominal e trocas entre grupos.

Esse bloco é a parte mais próxima de um raciocínio de projetista: identificar gargalos, mover massa de elementos pouco solicitados para elementos governantes e verificar se a solução melhorou após cada alteração.

### `planner_pipeline`

Controla quantos candidatos sobrevivem entre estágios, se os dois melhores seguem para detalhamento completo, profundidade de validação e tamanho do funil. Aumentar esses valores pode melhorar qualidade da busca, mas aumenta tempo computacional.

### `topology_cleanup`

Controla remoção de barras de baixa força, limiares de limpeza, preservação de simetria e segurança contra mecanismos. Essa limpeza só deve ser agressiva quando os critérios de reanálise estiverem ativos.

---

## Estrutura de diretórios e artefatos gerados

A pasta `outputs/` concentra os resultados. A execução pode limpar e recriar subpastas conforme configuração.

Artefatos importantes:

```text
outputs/config_requested.json                 Configuração solicitada.
outputs/config_used.json                      Configuração normalizada usada.
outputs/run_metadata.json                     Metadados da execução.
outputs/model/*.csv                           Nós, membros, apoios e cargas.
outputs/opensees/*.csv                        Resultados do solver interno.
outputs/frame3dd/*                            Arquivos de validação Frame3DD.
outputs/details/*                             Cortes, juntas, massa e montagem.
outputs/reports/relatorio_automatico.md       Relatório técnico automático.
outputs/final_report/*                        Relatório final consolidado.
outputs/plots/*                               Visualizações 2D/3D.
outputs/logs/planner_debug.jsonl              Log detalhado do planejador.
outputs/resultados_simulacao.zip              Pacote geral de resultados.
outputs/pacote_focado_fabricacao_e_calculo.zip Pacote focado em fabricação e cálculo.
```

Arquivos CSV permitem inspeção em planilha. Arquivos JSON preservam metadados e métricas estruturadas. Arquivos Markdown apresentam documentação interpretável. Os pacotes `.zip` consolidam os artefatos para compartilhamento.

---

## Execução pela interface Streamlit

A interface é iniciada por:

```bash
streamlit run app.py
```

A UI permite configurar parâmetros de ponte, materiais, carga, massa, objetivo, detalhamento e execução. O fluxo recomendado é preencher primeiro geometria e material, depois metas de resistência e massa, depois opções avançadas de busca. Após executar, analise primeiro o status de aprovação, depois a massa, depois o FS governante, depois a ruptura estimada e, por último, o detalhamento de juntas e cortes.

A UI é útil para iteração rápida porque permite alterar parâmetros sem editar JSON manualmente. Para reprodutibilidade rigorosa, preserve o `config_used.json` gerado em cada execução.

---

## Execução por CLI

A execução por CLI é definida em `run_cli.py`:

```bash
python run_cli.py
```

O script carrega `bridge_config.json`, executa `SimulationPipeline("outputs")` e imprime um resumo. Esse modo é adequado para repetição controlada, automação externa, comparação entre configurações e execução em ambiente sem interface gráfica.

Saída típica no terminal:

```text
Simulação concluída.
Status solver: ...
Frame3DD: ...
Menor FS membros principais: ...
Relatório: outputs/reports/relatorio_automatico.md
ZIP de resultados: outputs/resultados_simulacao.zip
```

---

## Testes automatizados

A suíte de testes está em `tests/`. Ela cobre normalização de configuração, lógica numérica central, segurança, arredondamento de corte, distribuição de carga, guarda de massa, planejamento de emendas, sizing, quarter-model, simetria, pacote de relatório, estimativa de ruptura, visualização e casos específicos de topologia.

Execute:

```bash
pytest -q
```

Testes relevantes por tema:

```text
test_config_normalization.py       Defaults e coerência de configuração.
test_core_numeric.py               Funções numéricas centrais.
test_core_safety.py                Condições de segurança.
test_load_distribution_service.py  Distribuição de cargas.
test_mass_guard.py                 Limite efetivo de massa.
test_connection_planner.py         Heurística de emendas.
test_quarter_model.py              Simetria e modelo reduzido.
test_cut_rounding.py               Arredondamento de cortes.
test_member_sizing_efficiency.py   Eficiência de reforço por massa.
test_rupture_estimator.py          Estimativa de ruptura.
test_pipeline_funnel.py            Integração do funil.
```

Sempre rode os testes após alterar fórmulas, critérios de aceitação, distribuição de cargas, semântica de massa ou detalhamento construtivo.

---

## Limitações científicas e hipóteses de modelagem

O solver interno é linear, elástico e axial. Ele não representa dano progressivo, plasticidade, abertura de juntas, escorregamento da cola, esmagamento local não linear, flambagem pós-crítica, grandes deslocamentos, contato real da placa de carga ou variabilidade estatística de materiais.

A flambagem é modelada por critérios analíticos simplificados. A interação axial-flexão por imperfeição é uma aproximação. A cola é avaliada por tensão média de cisalhamento, embora juntas reais possam falhar por peel, concentração de tensão, cura incompleta, espessura irregular ou aderência superficial ruim. A massa de cola depende fortemente do método de aplicação.

O Frame3DD, quando disponível, funciona como validação adicional, mas também depende da idealização do modelo. Divergências entre solver interno, Frame3DD e ensaio físico devem ser investigadas antes de confiar em uma solução.

A ruptura estimada por `P_projeto · FS_governante` supõe proporcionalidade linear. Essa hipótese é útil para comparação entre candidatos, mas não garante carga real de ruptura. O ensaio físico continua sendo o critério experimental decisivo.

---

## Checklist operacional recomendado

Antes de confiar em uma execução, verifique se as dimensões reais dos palitos foram medidas, se a massa por palito foi calibrada, se a resistência da cola representa o produto e método de cura usados, se o limite de massa corresponde à regra da competição, se todos os casos de carga relevantes estão ativados e se a solução final foi aprovada em S8.

Depois da execução, confira `config_used.json`, `final_validation_summary.json`, `relatorio_automatico.md`, `member_sizing_plan.csv`, `connection_plan.csv`, `stick_cut_list.csv`, `glue_joints.csv` e `mass_breakdown.csv`. Em seguida, inspecione visualmente os plots e procure membros com FS baixo, juntas críticas, apoios governantes, massa próxima demais do limite e emendas concentradas.

Durante a construção, não use apenas o comprimento geométrico do membro. Use a lista de corte e o plano de emendas. Respeite sobreposição, escalonamento, cura da cola, alinhamento, simetria e pressão de colagem. A maior parte das diferenças entre modelo e ensaio tende a surgir na execução física, não na matemática idealizada.

---

## Resumo técnico final

O projeto implementa um planejador ativo de ponte de palitos baseado em busca multiestágio, análise matricial de treliça 3D, checagem de flambagem, avaliação de massa e detalhamento construtivo. A lógica central é tratar a ponte como um sistema acoplado: geometria, força axial, estabilidade, massa, cola e montagem não são decisões independentes. Cada candidato só é aceito se atravessa o funil de domínio, solver, resistência, massa, robustez multi-loadcase, fabricação e validação final.

Em termos práticos, a automação responde a uma pergunta de projeto: dado um conjunto de palitos, cola, limites geométricos e meta de carga, qual configuração de treliça tem melhor chance computacional de entregar resistência suficiente sem ultrapassar massa e sem se tornar impossível de construir?
