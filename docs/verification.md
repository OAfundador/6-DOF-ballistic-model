# Prova de equivalência numérica

Saída literal de `python scripts/proof_of_equivalence.py`, gerada
automaticamente. Para regenerar:

```bash
python scripts/proof_of_equivalence.py --report docs/verification.md
```

O script sai com código 1 se qualquer verificação falhar, então este
arquivo só existe na forma abaixo se tudo passou.

```text
==============================================================================
PROVA DE EQUIVALÊNCIA — motor original vs pacote refatorado
==============================================================================

==============================================================================
AMBIENTE E PROCEDÊNCIA
==============================================================================

  data (UTC)     : 2026-09-07 03:32:54
  python         : 3.11.15 (Linux)
  numpy          : 2.4.4
  scipy          : 1.17.1
  pandas         : 3.0.2

  Arquivos congelados de referência (MD5):
    motor_original.py        9d1909cead60ace2eaf963b4b6493505
    apendice_dano.py         bff9266b9846ff6b5f6d75bf2fcece32
    legacy_motor.py          9fd38885ba132057cb408fa6057080d1

  Tabela fonte (origem)  : aero_coefficients_5in38.xlsx (1038fd7e7d00ea6ea36d1bf729ee8412)
  Coeficientes do modelo : aero_coefficients_5in38_spin73.npz (8d8ea820b1014b0ac7caf7fbd06abe39)

  Shim NumPy 2.x aplicado ao código congelado (tests/reference/compat.py):
  apenas a conversão de array 1x1 para float, que o NumPy 2.0 passou a
  rejeitar. Nenhum valor é alterado — ver tests/reference/README.md.

==============================================================================
1. TRAJETÓRIAS — comparação bit a bit do integrador
==============================================================================

Cada cenário é integrado pelos dois motores com os mesmos parâmetros.
`igual` usa numpy.array_equal sobre t e sobre o estado 12 x N inteiro:
igualdade exata, sem tolerância. O SHA-256 é dos bytes crus dos dois
arrays — se ele coincide, não há diferença em nenhum bit de nenhuma
das ~19 000 casas double de cada trajetória.

  Exemplo.py de referência (43.3°)
    amostras                    : 1671
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : fbf1a8a3961d3a024108a30a8ffa798d5c3b995bb781d0beb476e6bd8f87fcb7
    SHA-256 pacote refatorado   : fbf1a8a3961d3a024108a30a8ffa798d5c3b995bb781d0beb476e6bd8f87fcb7
    idêntico                    : SIM

  alcance máximo (39.6°, -1.35°)
    amostras                    : 1595
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : 2ed5f26601d944517bfdf443aad1983099a22df14be86ed2e6c4f25beeef477f
    SHA-256 pacote refatorado   : 2ed5f26601d944517bfdf443aad1983099a22df14be86ed2e6c4f25beeef477f
    idêntico                    : SIM

  meia elevação (20.0°, -1.00°)
    amostras                    : 1234
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : 8a87b78f5db8623f23d02f7bea7511fb2e51485fc8dfc37f135176bc14807abc
    SHA-256 pacote refatorado   : 8a87b78f5db8623f23d02f7bea7511fb2e51485fc8dfc37f135176bc14807abc
    idêntico                    : SIM

  tiro raso (5.0°, 0.00°)
    amostras                    : 751
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : ea1eae949345babbff0de44760cceb4c4e48c03424cf4cc4799ffbf9d39321b1
    SHA-256 pacote refatorado   : ea1eae949345babbff0de44760cceb4c4e48c03424cf4cc4799ffbf9d39321b1
    idêntico                    : SIM

  elevação negativa (-1.5°, -0.50°)
    amostras                    : 77
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : 8d840b37172be16f14179ad5b51c9c3ac9e2e23b6753ad3bc4e8edbcbb5f84fa
    SHA-256 pacote refatorado   : 8d840b37172be16f14179ad5b51c9c3ac9e2e23b6753ad3bc4e8edbcbb5f84fa
    idêntico                    : SIM

  limite do envelope (45.0°, -1.65°)
    amostras                    : 1658
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : c421c294aca0ef00d3947c68a8f33f18a2eba388dd1bda6013b2836bbb5e9384
    SHA-256 pacote refatorado   : c421c294aca0ef00d3947c68a8f33f18a2eba388dd1bda6013b2836bbb5e9384
    idêntico                    : SIM

  vento cruzado (35.0°)
    amostras                    : 1552
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : 1ddc8faf4650fa3479587a375e4d10819daad3d580470ec65e5d7dcec87fd451
    SHA-256 pacote refatorado   : 1ddc8faf4650fa3479587a375e4d10819daad3d580470ec65e5d7dcec87fd451
    idêntico                    : SIM

  plataforma em movimento (30.0°)
    amostras                    : 1421
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : d28c5d16dc7c56ccd132fead88bf755d7d9bdc9722b56d0ad373e00b9d9f7f42
    SHA-256 pacote refatorado   : d28c5d16dc7c56ccd132fead88bf755d7d9bdc9722b56d0ad373e00b9d9f7f42
    idêntico                    : SIM

  Resultado: 8/8 cenários idênticos bit a bit.

==============================================================================
2. GRANDEZAS DERIVADAS — velocidade, Mach, spin, ângulo de ataque
==============================================================================

O integrador é só metade da história: as grandezas derivadas são o que
aparece nos gráficos e nas estatísticas do TCC. Cenário de referência,
elevação 43.3°.

  |V| (m/s)                max |Δ| = 0.0e+00   idêntico: SIM
  Mach                     max |Δ| = 0.0e+00   idêntico: SIM
  |h| (rad/s)              max |Δ| = 0.0e+00   idêntico: SIM
  spin ω1 (rad/s)          max |Δ| = 0.0e+00   idêntico: SIM
  ângulo de ataque (°)     max |Δ| = 0.0e+00   idêntico: SIM

  Estatísticas de resumo (as quatro que o TCC cita):
    alcance (m)            original=16692.090536494743     refatorado=16692.090536494743     igual
    altura máxima (m)      original=5748.583283287444      refatorado=5748.583283287444      igual
    desvio lateral (m)     original=451.6897312915069      refatorado=451.6897312915069      igual
    tempo de voo (s)       original=66.04803979340255      refatorado=66.04803979340255      igual

==============================================================================
3. RELATÓRIO ANTIAÉREO — comparação caractere por caractere
==============================================================================

O `legacy_motor.py` congelado roda a main canônica do artigo; o pacote
refatorado produz o mesmo relatório pela API nova. As duas saídas são
comparadas linha a linha, do 'RESUMO DA TRAJETORIA' em diante.

  linhas comparadas           : 36
  SHA-256 legacy_motor.py     : a2298a9278da0765e12b55e5fc609f01c9b3d40a9536c2ae40798ed428ee3f5a
  SHA-256 pacote refatorado   : a2298a9278da0765e12b55e5fc609f01c9b3d40a9536c2ae40798ed428ee3f5a
  idêntico                    : SIM

  Relatório reproduzido (as duas saídas são este mesmo texto):

    | RESUMO DA TRAJETORIA
    | --------------------------------------------------------------------------------
    | Tempo de voo simulado : 61.226 s
    | Alcance final         : 16659.873 m
    | Altura maxima         : 5103.245 m
    | Desvio lateral maximo : 139.415 m
    | Velocidade final      : 317.175 m/s
    | Motivo da parada      : fuze
    | --------------------------------------------------------------------------------
    | RESUMO DO FUZE E DANO
    | --------------------------------------------------------------------------------
    | Fuze acionado dentro do raio : True
    | Origem do ponto de burst     : evento_integrador
    | Indice da amostra de burst   : 1585
    | Tempo do burst               : 61.226 s
    | Posicao do burst             : (16659.873, 220.510, -0.481) m
    | Distancia burst-alvo         : 24.380 m
    | Phi alvo (eixo i -> alvo)    : 1.664 deg
    | Alpha1 zona centro estatica  : 7.500 deg
    | Phi usado na velocidade      : 1.664 deg
    | Angulo velocidade -> alvo    : 1.771 deg
    | Zona polar NPG estatica      : (0.0, 15.0, 10)
    | Zona polar dinamica alpha2   : (0.000, 11.968) deg | hits NPG=10
    | Obliquidade no alvo          : 32.724 deg
    | Face dominante do alvo       : superior
    | Area exposta do alvo         : 4.151734 m^2
    | Contribuicoes de area        : {'superior': 3.6806158804221627, 'inferior': 0.0, 'lateral_direita': 0.0, 'lateral_esquerda': 0.0, 'traseira': 0.47111762624089076}
    | Velocidade inicial frag.     : 1560.668 m/s
    | Fragmentos totais no modelo  : 2113
    | Fragmentos na zona dinamica  : 17.362366
    | Area faixa dinamica          : 81.181222 m^2
    | Densidade rho''              : 0.213871706 frag/m^2
    | M fragmentos esperados       : 0.887938
    | Hipotese de penetracao       : todo fragmento que cruza a area penetra
    | Hipotese de dano             : qualquer penetracao e dano critico
    | Prob. destruicao Bernoulli   : 58.849674%

==============================================================================
4. MODELO DE DANO — contra o apêndice procedural congelado
==============================================================================

Três trajetórias sintéticas de geometria conhecida, cobrindo burst
frontal, eixo desalinhado da velocidade e burst oblíquo a 30°.

  burst frontal
    distância burst-alvo (m)         24.38                    == 24.38
    phi alvo (°)                     0.0                      == 0.0
    área exposta (m²)                4.375                    == 4.375
    velocidade do fragmento (m/s)    1863.6                   == 1863.6
    densidade (frag/m²)              0.3046462686469622       == 0.3046462686469622
    M fragmentos esperados           1.3328274253304597       == 1.3328274253304597
    P(destruição)                    0.7362694722439677       == 0.7362694722439677

  eixo != velocidade
    distância burst-alvo (m)         24.38                    == 24.38
    phi alvo (°)                     0.0                      == 0.0
    área exposta (m²)                4.375                    == 4.375
    velocidade do fragmento (m/s)    1863.6                   == 1863.6
    densidade (frag/m²)              0.3046462686469622       == 0.3046462686469622
    M fragmentos esperados           1.3328274253304597       == 1.3328274253304597
    P(destruição)                    0.7362694722439677       == 0.7362694722439677

  burst oblíquo 30°
    distância burst-alvo (m)         24.38                    == 24.38
    phi alvo (°)                     29.999999999999993       == 29.999999999999993
    área exposta (m²)                4.22636114155692         == 4.22636114155692
    velocidade do fragmento (m/s)    1807.3205466273675       == 1807.3205466273675
    densidade (frag/m²)              0.07808416110211394      == 0.07808416110211394
    M fragmentos esperados           0.3300118642530447       == 0.3300118642530447
    P(destruição)                    0.28108479601056924      == 0.28108479601056924

  Resultado: todos os valores coincidem exatamente.

==============================================================================
5. SELEÇÃO DE PONTOS DE MIRA — contra a tabela publicada
==============================================================================

A seleção refatorada roda sobre a tabela de azimutes ótimos publicada
e o resultado é comparado com a tabela de pontos de mira usada no TCC.

  entrada  : optimal_azimuths_zero_drift.xlsx (601 elevações)
  publicado: selected_points_100m.xlsx (163 pontos)
  reproduzido: 163 pontos

    Elevacao_deg               idêntico coluna inteira: SIM
    Azimute_otimo_deg          idêntico coluna inteira: SIM
    Alcance_x_m                idêntico coluna inteira: SIM
    Desvio_z_resultante_m      idêntico coluna inteira: SIM

  Primeiras cinco linhas (publicado | reproduzido):
    elev   39.6° azim  -1.35° alcance 16796.794263 m  |  elev   39.6° azim  -1.35° alcance 16796.794263 m
    elev   36.2° azim  -1.20° alcance 16715.886338 m  |  elev   36.2° azim  -1.20° alcance 16715.886338 m
    elev   34.8° azim  -1.15° alcance 16634.563480 m  |  elev   34.8° azim  -1.15° alcance 16634.563480 m
    elev   33.7° azim  -1.10° alcance 16550.861220 m  |  elev   33.7° azim  -1.10° alcance 16550.861220 m
    elev   32.8° azim  -1.10° alcance 16469.309281 m  |  elev   32.8° azim  -1.10° alcance 16469.309281 m

==============================================================================
VEREDITO
==============================================================================

  trajetórias              OK — idêntico
  grandezas derivadas      OK — idêntico
  relatório antiaéreo      OK — idêntico
  modelo de dano           OK — idêntico
  seleção de pontos        OK — idêntico

  TODAS AS VERIFICAÇÕES PASSARAM.

```
