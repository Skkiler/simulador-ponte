# Relatório automático da simulação

## Configuração analisada

- Vão livre: 1200 mm
- Largura: 170 mm
- Altura central: 220 mm
- Carga aplicada: 90.00 kgf
- Módulo de elasticidade adotado: 6000.0 MPa

## Métricas principais

- Nós: 60
- Membros: 234
- Apoios ativos: 4
- Apoios com perda de contato: 4
- Erro de equilíbrio vertical: 2.956e-12 N
- Menor FS em membros principais: 0.5367807901155514
- Menor FS em todos os membros: 0.06899515665510557

## Análise automática

Solver axial: regular. Erro de equilíbrio vertical: 2.956e-12 N.
Há 32 membros principais com FS < 1,0. O projeto precisa de redimensionamento antes de considerar a carga segura.
42 estabilizadores aparecem comprimidos/esbeltos. Interprete-os como travamento/tension-only ou reforce-os se forem usados como barras comprimidas reais.
4 pontos de apoio perderam contato no modelo unilateral. Isso é coerente com apoio livre, mas concentra reação nos apoios internos.
2 apoios ativos excedem a capacidade simplificada adotada. Reforce a região de apoio e aumente área de contato.
Modelo peça-a-peça: 936 palitos estimados com perdas e massa total de 1329.2 g.
A massa estimada excede o limite configurado; reduza reforços ou revise a geometria.

## Sugestões de melhoria

- Membro 108 (vertical, FS=0.54): crítico por buckling_y. Aumente a inércia da seção, reduza o comprimento livre ou adicione travamento intermediário.
- Membro 35 (vertical, FS=0.54): crítico por buckling_y. Aumente a inércia da seção, reduza o comprimento livre ou adicione travamento intermediário.
- Membro 14 (top_chord, FS=0.56): crítico por compressão direta. Aumente a área efetiva ou distribua melhor o esforço.
- Membro 87 (top_chord, FS=0.56): crítico por compressão direta. Aumente a área efetiva ou distribua melhor o esforço.
- Membro 16 (top_chord, FS=0.57): crítico por compressão direta. Aumente a área efetiva ou distribua melhor o esforço.
- Membro 85 (top_chord, FS=0.57): crítico por compressão direta. Aumente a área efetiva ou distribua melhor o esforço.
- Membro 12 (top_chord, FS=0.58): crítico por compressão direta. Aumente a área efetiva ou distribua melhor o esforço.
- Membro 89 (top_chord, FS=0.58): crítico por compressão direta. Aumente a área efetiva ou distribua melhor o esforço.
- Membro 106 (vertical, FS=0.59): crítico por buckling_y. Aumente a inércia da seção, reduza o comprimento livre ou adicione travamento intermediário.
- Membro 37 (vertical, FS=0.59): crítico por buckling_y. Aumente a inércia da seção, reduza o comprimento livre ou adicione travamento intermediário.
- Apoios: aumente o grupo support_pad, acrescente travessas inferiores em x=0 e x=span, ou distribua o contato em área maior.
- Contraventamentos: mantenha X duplo, mas trate o par como tension-only; se trabalhar comprimido de verdade, use dois palitos ou reduza o vão livre.
- Membro 4 (top_chord, FS=1.18): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 6 (top_chord, FS=0.80): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 8 (top_chord, FS=0.68): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 10 (top_chord, FS=0.62): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 12 (top_chord, FS=0.58): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 14 (top_chord, FS=0.56): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 16 (top_chord, FS=0.57): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 18 (top_chord, FS=0.60): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 20 (top_chord, FS=0.65): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 22 (top_chord, FS=0.74): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 24 (top_chord, FS=0.90): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Membro 26 (top_chord, FS=1.38): reforçar: adicionar palitos contínuos ou aumentar sobreposição/talas
- Massa: margem pequena. Procure reforços removíveis em estabilizadores e membros de baixa solicitação.
- Proposta automática: testar Parker / parker_plateau com altura 187 mm e painel 100 mm. Use outputs/optimization/recommended_config.json.


## Modelo peça-a-peça e massa

- Palitos estimados com perdas: 936
- Peças individuais: 952
- Palitos brutos antes de perdas: 866
- Área total de cola estimada: 76440.0 mm²
- Massa de cola estimada: 18.8 g
- Massa total estimada: 1329.2 g
- Margem de massa: -329.2 g
- Resistência de cisalhamento da cola adotada: 3.50 MPa

Arquivos detalhados: `outputs/details/stick_pieces.csv`, `glue_joints.csv`, `cutting_list.csv`, `blank_cut_plan.csv`, `member_detail_checks.csv`, `reinforcement_suggestions.csv`.

## Imagens geradas

Veja a pasta `outputs/plots/`.

## Observações

Este relatório é preliminar. O modelo axial em NumPy calcula forças axiais, reações e deslocamentos sob hipótese linear. Flambagem é verificada por Euler. O detalhamento peça-a-peça estima cortes, emendas e cola; não substitui ensaio físico nem FEM de contato/cola.
