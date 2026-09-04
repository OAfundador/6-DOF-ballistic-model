# Reprodução da campanha Monte Carlo do TCC

Saída literal de `scripts/reproduce_campaign_point.py`.

```text
==============================================================================
REPRODUÇÃO DA CAMPANHA MONTE CARLO DO TCC (tiro contra drone naval)
==============================================================================

  data (UTC)        : 2026-09-06 15:58:45
  numpy / scipy     : 2.4.4 / 1.17.1
  campanha publicada: monte_carlo_campanha_publicada.xlsx (163 pontos)
  pontos de mira    : selected_points_100m.xlsx (163 pontos)
  disparos por ponto: 1000
  seed              : 16184331
  perturbações      : elevação N(0, 0.1°), azimute N(0, 0.05°)

  As perturbações são sorteadas para a campanha inteira (163 pontos) e
  fatiadas por ponto, porque o gerador legado sorteia todas as elevações
  antes de todos os azimutes — sortear só para um ponto daria azimutes
  diferentes.

------------------------------------------------------------------------------
  PONTO 162 — elevação -1.1°, azimute -0.05°, alcance nominal 430.130 m

    1000 tiros em 146.1 s

    contagem de acertos               publicado  reproduzido   igual
    Acertos_Drone_Sea_Baby                   78           78   SIM
    Acertos_IRIS_Paykan                     715          715   SIM
    Acertos_Osa_class                       522          522   SIM
    Acertos_Hayabusa_class                  663          663   SIM
    Acertos_SMS_V4                          826          826   SIM
    Acertos_PT_105                          351          351   SIM

    estatística contínua                publicado    reproduzido      Δ rel
    Erro_X_medio_m                    1.749706364    1.749704802   8.93e-07
    Erro_X_std_m                     26.915517342   26.915513489   1.43e-07
    Erro_X_min_m                    -64.091008842  -64.091012566   5.81e-08
    Erro_X_max_m                    114.825967894  114.825967892   2.20e-11
    Erro_Z_medio_m                   -0.021718596   -0.021718573   1.06e-06
    Erro_Z_std_m                      0.390650325    0.390650339   3.53e-08
    Erro_Z_min_m                     -1.302076557   -1.302075965   4.55e-07
    Erro_Z_max_m                      1.056996182    1.056995893   2.73e-07
    CEP50_m                          18.351931836   18.351951558   1.07e-06
    CEP90_m                          44.326492959   44.326479287   3.08e-07
    CEP95_m                          52.062928563   52.062927821   1.42e-08
    Tempo_voo_medio_s                 0.545387656    0.545387654   3.70e-09

    acertos idênticos          : SIM
    Δ relativa máxima (stats)  : 1.07e-06  (limite 1e-04)
    Δ absoluta máxima (stats)  : 0.020 mm

------------------------------------------------------------------------------
  PONTO 163 — elevação -1.5°, azimute -0.05°, alcance nominal 340.905 m

    1000 tiros em 123.9 s

    contagem de acertos               publicado  reproduzido   igual
    Acertos_Drone_Sea_Baby                  115          115   SIM
    Acertos_IRIS_Paykan                     881          881   SIM
    Acertos_Osa_class                       717          717   SIM
    Acertos_Hayabusa_class                  834          834   SIM
    Acertos_SMS_V4                          957          957   SIM
    Acertos_PT_105                          491          491   SIM

    estatística contínua                publicado    reproduzido      Δ rel
    Erro_X_medio_m                    0.706482146    0.706481815   4.69e-07
    Erro_X_std_m                     18.119794419   18.119793756   3.66e-08
    Erro_X_min_m                    -54.245150934  -54.245151200   4.92e-09
    Erro_X_max_m                     71.021058000   71.021058432   6.08e-09
    Erro_Z_medio_m                    0.010084044    0.010084046   2.13e-07
    Erro_Z_std_m                      0.305096320    0.305096312   2.56e-08
    Erro_Z_min_m                     -0.926866727   -0.926866811   9.01e-08
    Erro_Z_max_m                      0.886156005    0.886155931   8.40e-08
    CEP50_m                          12.372787713   12.372795566   6.35e-07
    CEP90_m                          29.848851003   29.848851016   4.19e-10
    CEP95_m                          35.236077238   35.236077551   8.87e-09
    Tempo_voo_medio_s                 0.429760882    0.429760881   9.86e-10

    acertos idênticos          : SIM
    Δ relativa máxima (stats)  : 6.35e-07  (limite 1e-04)
    Δ absoluta máxima (stats)  : 0.008 mm

==============================================================================
VEREDITO
==============================================================================

  Contagens de acertos idênticas e estatísticas dentro do limite.

```
