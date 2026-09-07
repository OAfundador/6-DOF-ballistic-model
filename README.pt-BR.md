# Modelo Balístico 6-DOF

Modelo de balística exterior com seis graus de liberdade para projéteis
estabilizados por rotação, seguindo R. McCoy, *Modern Exterior Ballistics: The
Launch and Flight Dynamics of Symmetric Projectiles* (2ª ed.).

Desenvolvido para o Trabalho de Conclusão de Curso em Matemática Aplicada e
Computacional no **IME-USP** (Instituto de Matemática e Estatística,
Universidade de São Paulo), com parâmetros ajustados para um caso de artilharia
naval: o canhão de calibre 5"/38.

*(Read in [English](README.md).)*

---

## Qual versão você está vendo

> **Esta branch é uma refatoração, não o artefato do TCC.**
>
> O código que o autor escreveu, verificou e usou no TCC é o motor de arquivo
> único da branch [`legacy`](../../tree/legacy). É dele que saiu cada número do
> trabalho escrito.
>
> Esta branch `main` é aquele mesmo motor reorganizado em pacote. Foi escrita
> pelo **Claude** (assistente de IA da Anthropic) a pedido do autor, e a suíte
> de testes que a verifica também foi escrita e executada pelo Claude. O autor
> revisou e aprovou o resultado, mas não refez os números do TCC a partir dela.
>
> A refatoração é **bit a bit idêntica** ao original, e isso é demonstrado, não
> afirmado — veja [Verificação](#verificação) e
> [`docs/verification.md`](docs/verification.md), que é a saída literal de um
> script que você mesmo pode rodar. Ainda assim:
>
> - **Vai citar ou auditar o TCC?** Use a `legacy`. É o artefato que o autor
>   verificou.
> - **Vai construir em cima do modelo?** Use a `main`. Mesma física, mesmos
>   números, num formato que dá para estender.

---

## O que há aqui

Três camadas, cada uma utilizável isoladamente:

| Camada | Módulo | O que faz |
| --- | --- | --- |
| **Núcleo** | `sixdof` | Integra as equações de movimento 6-DOF de um tiro: arrasto, sustentação, Magnus, momento de tombamento e amortecimentos, com coeficientes interpolados em Mach e ângulo de ataque total. |
| **Antiaérea** | `sixdof.aa` | Espoleta de proximidade, geometria de alvo por faces, ogiva fragmentária com distribuição polar de hits e modelo de dano que devolve a probabilidade de destruição. |
| **Monte Carlo** | `sixdof.montecarlo` | A campanha de dispersão completa do TCC: varredura angular, seleção de pontos de mira, tiro perturbado com contagem de acertos e custo esperado de engajamento. |

As camadas antiaérea e de Monte Carlo são opcionais — importar `sixdof` não
carrega nenhuma delas.

---

## Começando

```bash
git clone https://github.com/OAfundador/6-DOF-ballistic-model.git
cd 6-DOF-ballistic-model
pip install -r requirements.txt

python examples/01_single_shot.py          # uma trajetória + 18 gráficos
python examples/02_aa_engagement.py        # espoleta + dano fragmentário
```

Ou instalando o pacote:

```bash
pip install -e ".[dev]"
pytest
```

Python 3.9 ou superior. As dependências obrigatórias são NumPy, SciPy, pandas e
openpyxl; matplotlib só é necessário para os gráficos.

---

## Uso como biblioteca

```python
from sixdof import (
    BallisticSimulator, naval_5in38_coefficients,
    naval_5in38_gun, naval_5in38_projectile, standard_atmosphere,
)

simulador = BallisticSimulator(
    projectile=naval_5in38_projectile(),
    weapon=naval_5in38_gun(elevation_deg=43.3, azimuth_deg=0.0),
    environment=standard_atmosphere(),
    aero_coeffs=naval_5in38_coefficients(),
)

trajetoria = simulador.simulate(verbose=False)
print(trajetoria.max_range, trajetoria.max_altitude, trajetoria.flight_time)
```

Os atributos em português do motor antigo (`alcance_max`, `altura_max`,
`desvio_lateral_max`, `tempo_voo`) continuam funcionando como apelidos, assim
como o nome `RealAerodynamicCoefficients`.

Cada peça pode ser trocada — outro projétil, plataforma em movimento, vento,
outra tabela de coeficientes:

```python
from sixdof import Environment, Projectile, Vessel, Weapon

contratorpedeiro = Vessel("DD", center_position=(0, 0), length=115, width=12,
                          height=10, velocity=(12.0, 0.0))
arma = Weapon(position=(20, 8, 0), elevation_deg=30, azimuth_deg=-1.2,
              mounted_on_vessel=contratorpedeiro)
ambiente = Environment(rho=1.18, W1=-8.0, W3=3.0)
```

### A atmosfera: uniforme por padrão, em camadas se você pedir

O `Environment` guarda uma densidade e uma velocidade do som para o voo inteiro.
Foi o que o TCC assumiu, e continua sendo o padrão para que todo número que
este repositório reproduz siga reproduzido. Também é uma hipótese de verdade, e
vale saber quando ela começa a custar: um tiro naval que sobe a menos de 3 km
quase não sente, enquanto um obus que chega a 5,6 km atravessa ar uns 40% mais
rarefeito e com som 7% mais lento, o que muda o alcance em cerca de 12%.

Para tiros com teto de verdade, o `LayeredAtmosphere` é o modelo ICAO —
troposfera caindo 6,5 K/km até 11 km, isotérmica acima:

```python
from sixdof.environment import LayeredAtmosphere

ambiente = LayeredAtmosphere()      # 1,225 kg/m³ e 340,29 m/s ao nível do mar
ambiente.density_at(5000.0)         # 0,7361
ambiente.sound_speed_at(5000.0)     # 320,53
```

O motor **pergunta** `density_at(h)` e `sound_speed_at(h)` ao ambiente em vez de
ler os atributos, e a classe base responde as duas com as constantes que já
tinha. Então trocar a classe é a mudança inteira, um perfil próprio são dois
métodos, e o caso uniforme fica intocado — a prova de equivalência bit a bit
cobre isso.

A velocidade do som pesa mais do que parece, porque é ela que dá o Mach que
indexa a tabela de coeficientes. Travada nos 340 m/s do nível do mar, um tiro
alto lê seus coeficientes com até 13% de erro em Mach bem na faixa transônica,
onde o arrasto varia por um fator 2,5.

### Engajamento antiaéreo

```python
from sixdof.aa import ProximityFuze, evaluate_engagement, shahed_136, vt_fcl_mk49

alvo = shahed_136(center=(16673.0, 200.0, 0.7))
ogiva = vt_fcl_mk49()
espoleta = ProximityFuze(target_center=alvo.center, radius_m=24.38, arm_time_s=0.5)

trajetoria = simulador.simulate(fuze=espoleta)   # a integração para no burst
burst, dano = evaluate_engagement(trajetoria, alvo, ogiva, espoleta)

print(dano.expected_fragments, dano.p_destruction)
```

Nada nessa cadeia é específico do Shahed ou da munição VT. Um alvo é uma lista
de faces com área e normal; uma ogiva é uma velocidade de ejeção mais uma tabela
de zonas polares. Ambos têm construtores genéricos:

```python
from sixdof.aa import FragmentationWarhead, PolarZone, box_target, triangular_prism_target

alvo = box_target("quadricoptero", length=0.6, width=0.6, height=0.2,
                  center=(4000.0, 150.0, 0.0))
ogiva = FragmentationWarhead(
    name="espoleta de proximidade 40 mm",
    fragment_velocity_mps=1100.0,
    polar_zones=[PolarZone(0, 90, 300), PolarZone(90, 180, 100)],
    effective_fragments=400,
)
```

### Coeficientes: os sete que a equação lê

O modelo lê **sete** números, e nada além disso:

| Entrada | Termo na equação | Depende de |
| --- | --- | --- |
| `CD` | força de arrasto, `C_D` | Mach e ângulo de ataque |
| `CLA` | força de sustentação, `C_Lα` | Mach e ângulo de ataque |
| `CNP` | momento de Magnus, `C_Mpα` | Mach e ângulo de ataque |
| `CYP` | força de Magnus, `C_Npα` | Mach |
| `CLP` | amortecimento de spin, `C_lp` | Mach |
| `CMA` | momento de tombamento, `C_Mα` | Mach |
| `CMQ` | amortecimento de arfagem, `C_Mq` | Mach |

O `AerodynamicCoefficients` guarda exatamente esses, na forma que você tiver —
constante, tabela em Mach, grid em `(Mach, ângulo)` ou função. O que for omitido
vale zero, que é como se desliga um termo:

```python
from sixdof import AerodynamicCoefficients, load_coefficients

AerodynamicCoefficients(CD=0.3, CLA=1.8, CMA=3.5, CMQ=-9.4, CLP=-0.03)
AerodynamicCoefficients(mach_grid=machs, CD=valores_cd, CLA=valores_cla, ...)
AerodynamicCoefficients(CD=lambda mach, alpha: 0.2 + 0.5 * np.sin(alpha) ** 2)

load_coefficients("meu_projetil.npz")   # ou .xlsx, ou .csv — ele lê o conteúdo
```

#### A referência na entrada é o McCoy

Os sete são lidos como o **McCoy, *Modern Exterior Ballistics*, 2ª ed., cap. 2**
os define, e o `sixdof.dynamics` implementa esse capítulo termo a termo. Duas
propriedades dessa referência decidem o que os seus números têm de significar, e
nenhuma delas é visível no nome de uma coluna:

**Adimensionalização.** Os quatro que multiplicam uma velocidade angular —
`CLP`, `CMQ`, `CYP`, `CNP` — são adimensionalizados em `(pd/V)`, conforme a
eq. (2.4) do McCoy. O sistema aerobalístico NACA usa `(pd/2V)`, e o próprio
McCoy avisa da consequência: *"uma diferença de fator dois nos coeficientes que
dependem de velocidade angular"*. Uma tabela normalizada em NACA precisa desses
quatro **divididos por 2** na entrada. O `CMA` não carrega velocidade angular e é
idêntico nos dois sistemas — é a verificação cruzada.

**Sistema de eixos e sinal.** `CD` e `CLA` são coeficientes de eixo-vento. Uma
fonte que dá força axial e normal em eixo-corpo precisa da rotação pelo ângulo
de ataque antes, e o sinal do termo axial depende de a fonte contá-lo positivo
para frente (convenção do McCoy, `C_X ≈ −C_D`) ou positivo para trás.

Nenhum dos dois erros levanta exceção. Os dois produzem uma trajetória plausível
e uma resposta errada.

#### Converter a tabela da fonte é tarefa sua

Relatórios de túnel de vento, redução de provas de tiro, CFD e cada código de
predição tabulam suas próprias grandezas intermediárias — força axial separada
em `CX0` e `CX2`, força normal como `CNA`, Magnus como série em `CNPA`/`CNPA3` —
e cada convenção exige aritmética própria para chegar nos sete. **Essa aritmética
não está no pacote.** Conversão enterrada em biblioteca é conversão que ninguém
confere, e as duas armadilhas acima são exatamente do tipo que passa pela
revisão. Converta sua tabela uma vez, ao lado dos seus dados, onde dá para ler, e
entregue os sete.

O `tests/test_coefficients.py` cobra isso do pacote: tokeniza **todos** os
módulos em `src/sixdof/`, joga fora comentários e strings, e exige que nenhum
nome de coluna de convenção de fonte apareça em código executável em lugar
nenhum. O `load_coefficients` lê os sete — de `.npz`, de uma planilha de duas
abas, ou de uma tabela `Mach` + sete colunas — e recusa qualquer outra coisa em
vez de adivinhar.

O `examples/07_bring_your_own_table.py` é a conversão trabalhada para copiar.

#### A tabela do 5"/38 que vem junto

- `data/aero_coefficients_5in38_spin73.npz` — os sete, no grid `(Mach, ângulo)`
  completo. É o padrão que o `naval_5in38_coefficients()` carrega, e a tabela em
  que a campanha do TCC foi voada.
- `data/aero_coefficients_5in38_spin73_sheets.xlsx` — os mesmos sete em duas
  abas editáveis (`mach_only` e `yaw_dependent`).
- `data/aero_coefficients_5in38.xlsx` — a tabela fonte, mantida por procedência.
  O pacote não a lê; o exemplo e os testes congelados leem.

O nome diz `spin73` porque **esta tabela não cumpre o contrato do McCoy acima**,
de duas maneiras conhecidas — as duas herdadas do TCC, as duas mantidas de
propósito, para que os resultados publicados continuem reprodutíveis:

1. `CLP`, `CMQ`, `CYP` e `CNP` vêm de uma tabulação SPIN-73, normalizada em
   `(pd/2V)`, logo estão no dobro do que a equação quer. Sintoma: spin no
   impacto de 94 rev/s onde um código independente dá 129.
2. O `CD` foi montado subtraindo o termo de guinada em vez de somar, então o
   arrasto cai com o ângulo de ataque em vez de subir. Vale cerca de −0,3 % de
   alcance na trajetória do TCC, e mais em guinada alta.

O [`docs/table_5in38_provenance.md`](docs/table_5in38_provenance.md) traz a
derivação, as fontes primárias, as medições e como rodar a física corrigida.

**Uma ressalva antes de achatar mais.** É tentador jogar fora o eixo do ângulo
de ataque e ficar com uma tabela `Mach, CD, CLA, CYP, CLP, CMA, CNP, CMQ`. `CD`
e `CLA` sobrevivem razoavelmente — variam alguns por cento na faixa de ângulo
que um projétil estável realmente voa. `CNP` não: o momento de Magnus é **ímpar**
no ângulo de ataque, ou seja, é exatamente zero em ângulo nulo, e nenhum valor
único indexado por Mach pode representá-lo. Amostrar em α=0 não aproxima o
termo, apaga. Medido no tiro de referência, a tabela só-Mach em α=0 move o
alcance em 12,6 m nos 16,7 km e a deriva em 0,9 m nos 452 m — pouco, mas não
nada, e o `from_mach_table` avisa isso na própria docstring.

### Campanha de Monte Carlo

```python
from sixdof import surface_target_fleet
from sixdof.montecarlo import AimPoint, DispersionSettings, MonteCarloCampaign

campanha = MonteCarloCampaign(
    simulador, surface_target_fleet,
    DispersionSettings(n_shots=1000, sigma_elevation_deg=0.1, sigma_azimuth_deg=0.05),
)
resultados = campanha.run([AimPoint(39.6, -1.35, 16796.8, 4.26)])
tabela = MonteCarloCampaign.to_frame(resultados)
```

---

## Estrutura

```
src/sixdof/
  aerodynamics.py   os sete coeficientes que a equação lê, e o carregador
  projectile.py     massa, inércias, calibre, raiamento
  weapon.py         reparo, ângulos de tiro, acoplamento com a plataforma
  vessel.py         alvo de superfície em forma de caixa
  environment.py    densidade, gravidade, vento, velocidade do som
  dynamics.py       equações de movimento e estado inicial
  events.py         condições de parada: solo e espoleta de proximidade
  simulator.py      driver de integração
  trajectory.py     históricos de estado, grandezas derivadas, estatísticas
  plotting.py       os dezoito gráficos padrão
  presets.py        a configuração 5"/38 usada em todo o TCC
  aa/               geometria do alvo, ogiva, espoleta, modelo de dano
  montecarlo/       varredura, seleção de pontos, dispersão, custo

examples/           sete scripts executáveis -- ver examples/README.md
scripts/            proof_of_equivalence.py, reproduce_campaign_point.py
tests/              a suíte, incluindo a regressão bit a bit
tests/reference/    cópias congeladas do código pré-refatoração (nunca importadas)
data/               tabela de coeficientes e resultados intermediários publicados
docs/               arquitetura, procedência da tabela, geometria do alvo,
                    saída da verificação
```

---

## Verificação

O motor de onde este repositório saiu era um único arquivo de 1600 linhas
(`Motor.py`), cujos resultados estão citados no TCC. A refatoração é **bit a
bit idêntica** a ele, e isso é demonstrado, não afirmado.

`tests/reference/` guarda cópias inalteradas do código pré-refatoração — o
motor de arquivo único, o motor antiaéreo e o apêndice de dano — com os MD5
documentados em [`tests/reference/README.md`](tests/reference/README.md), para
você conferir que são mesmo os bytes originais. Nada em `src/` importa esses
arquivos; eles existem só para servir de comparação.

### Rode a prova você mesmo

```bash
python scripts/proof_of_equivalence.py
```

O script carrega o código congelado e o pacote, roda os dois lado a lado e sai
com código 1 se qualquer coisa divergir. A saída armazenada é
[`docs/verification.md`](docs/verification.md), regerada com
`--report docs/verification.md`.

Para cada um de oito cenários ele reporta o número de amostras, a diferença
absoluta máxima sobre o histórico de estado `12 × N` inteiro e um **SHA-256 dos
bytes crus de cada motor**. Digests iguais significam que não há um bit
diferente em nenhum dos ~19 000 doubles de cada trajetória:

```text
  Exemplo.py de referência (43.3°)
    amostras                    : 1671
    max |Δ| em todo o estado    : 0.0e+00
    SHA-256 motor original      : fbf1a8a3961d3a024108a30a8ffa798d5c3b995bb781d0beb476e6bd8f87fcb7
    SHA-256 pacote refatorado   : fbf1a8a3961d3a024108a30a8ffa798d5c3b995bb781d0beb476e6bd8f87fcb7
    idêntico                    : SIM
```

### O que é conferido

| Verificação | Contra o quê | Resultado |
| --- | --- | --- |
| Grids de coeficientes e consultas pontuais | `motor_original.py` congelado | idêntico |
| Estado inicial e lado direito das equações | `motor_original.py` congelado | idêntico |
| Trajetórias inteiras, 8 cenários (5 elevações, vento, plataforma em movimento) | `motor_original.py` congelado | idêntico, SHA-256 igual |
| Históricos derivados — velocidade, Mach, `\|h\|`, spin, ângulo de ataque | `motor_original.py` congelado | max \|Δ\| = 0 |
| As quatro estatísticas que o TCC cita | `motor_original.py` congelado | iguais como float |
| Relatório antiaéreo de console | `legacy_motor.py` congelado | caractere por caractere |
| Modelo de dano, 3 geometrias de burst | `apendice_dano.py` congelado | todos os valores iguais |

### Uma conferência de fora: um caso M107 publicado

Tudo acima compara o pacote com o motor de onde ele veio. Isso prova que a
refatoração é fiel. Não prova que a física está certa, porque os dois lados
compartilham as equações, os dados e qualquer erro em um dos dois.

Então um exemplo voa o projétil de outra pessoa com os coeficientes de outra
pessoa:

```bash
python examples/11_m107_benchmark.py
```

Ele pega o caso do obus 155 mm M107 e a Tabela 1 de

> Khalil, M., Abdalla, H., Kamal, O., "Dispersion Analysis for Spinning
> Artillery Projectile", ASAT-13, Military Technical College, Cairo, 2009

e compara com os números que o próprio artigo declara no texto. Como segunda opinião, cita também o
**[RigidFlightLab](https://github.com/timeout187/RigidFlightLab)** — vale a pena
dar uma olhada no trabalho dele, fica a recomendação: é uma implementação aberta
e independente do mesmo caso em outra formulação, referencial não-rolante,
`[x,y,z,u,v,w,φ,θ,ψ,p,q,r]`, RK45, enquanto este pacote usa a forma vetorial do
McCoy. Dois códigos que não compartilham uma linha de fonte nem o integrador,
concordando sobre o projétil de um terceiro, é uma afirmação mais forte do que
qualquer um dos dois faz sozinho — então este benchmark existe porque aquele
projeto existe.

| Grandeza | Artigo | Código indep. | Este pacote |
| --- | --- | --- | --- |
| Tempo de voo (s) | 66,67 | 66,40 | 66,137 |
| Tempo ao apogeu (s) | 31,00 | 30,50 | 30,386 |
| Desaceleração axial inicial (g) | −4,45 | −4,47 | −4,468 |
| Ângulo de ataque máximo (°) | 1,29 | 1,30 | 1,287 |
| Apogeu (m) | ~5600 | 5647 | 5635,7 |
| Spin no impacto (rev/s) | — | 128,8 | 128,853 |
| Deriva (m) | — | 483 | 482,821 |

Dentro de 0,8% do artigo em tudo que ele publica como texto, e dentro de 0,5% do
código independente. Os dois códigos ficam ~1,7% abaixo do artigo no apogeu,
provavelmente porque o artigo usa uma atmosfera própria não especificada e
inclui termos de Coriolis que nenhum dos dois implementa.

Chegar lá exige duas conversões na entrada, e o exemplo foi escrito para deixar
as duas visíveis em vez de silenciosamente certas. A Tabela 1 é adimensionalizada
em `(pd/2V)` enquanto as equações do McCoy usam `(pd/V)`, então os quatro
coeficientes dependentes de velocidade angular são divididos por 2 — o `CMA` não
carrega taxa e passa ileso, que é a conferência cruzada que distingue diferença
de convenção de tabela errada. E a Tabela 1 dá coeficientes de corpo, `C_A` e
`C_Nα`, conforme a Nomenclature do próprio artigo, então precisam da rotação pelo
ângulo de ataque descrita em
[a tabela do 5"/38 que vem junto](#a-tabela-do-538-que-vem-junto). O script
imprime a rodada com e sem cada hipótese, de modo que o preço delas é medido em
vez de afirmado — só o fator 2 vale 61 m de deriva e 34 rev/s de spin.

### A campanha contra drone naval — o caso do TCC propriamente dito

A camada antiaérea acima é trabalho novo. O que o TCC trata de fato é tiro de
superfície contra drones navais, e isso passa pelas quatro etapas de Monte
Carlo. Cada uma é conferida contra o que o TCC publicou:

| Etapa | Verificação | Resultado |
| --- | --- | --- |
| 1 — varredura angular | Linhas da varredura publicada reintegradas | Bate em ~1e-9 relativo, sempre abaixo do milímetro — ver a ressalva |
| 2 — seleção de pontos | Os 163 pontos de mira publicados | Reproduzidos célula a célula |
| 3 — campanha de dispersão | Contagens de acerto publicadas nos pontos 1, 160, 162 e 163 (1000 tiros cada) | **Todas as contagens exatas**; estatísticas de dispersão em ~1e-6 relativo, pior caso 0,4 mm |
| 4 — custo de engajamento | `Custo.py` congelado sobre as taxas publicadas | Todos os campos iguais, e a curva E[custo] inteira |

O ponto 1 é a solução de alcance máximo com que o TCC abre. Refazendo os 1000
tiros dele, as seis contagens de acerto foram reproduzidas exatamente — 63,
227, 227, 249, 219, 182 — e portanto as seis taxas publicadas:

```text
    Acertos_Drone_Sea_Baby                   63           63   SIM
    Acertos_IRIS_Paykan                     227          227   SIM
    Acertos_Osa_class                       227          227   SIM
    Acertos_Hayabusa_class                  249          249   SIM
    Acertos_SMS_V4                          219          219   SIM
    Acertos_PT_105                          182          182   SIM
```

Reproduza você mesmo:

```bash
python scripts/reproduce_campaign_point.py              # pontos 162 e 163, minutos
python scripts/reproduce_campaign_point.py --point 1    # alcance máximo, ~1 hora
```

A saída armazenada é [`docs/verification_campaign.md`](docs/verification_campaign.md),
cobrindo os pontos 162 e 163; os pontos 1 e 160 foram verificados do mesmo jeito e
estão na tabela acima.

### Uma ressalva sobre as tabelas publicadas

As planilhas publicadas **não** são reproduzíveis bit a bit em outra máquina, e
isso é característica do código original, não desta refatoração. O motor
congelado e este pacote batem exatamente entre si, enquanto **os dois** diferem
da planilha pela mesma quantidade minúscula — tipicamente 1e-9 relativo, pior
caso ~1e-7, nunca mais que um milímetro de alcance.

A causa é banal: o `solve_ivp` é adaptativo, as planilhas foram geradas no
Windows com SciPy/NumPy mais antigos, e uma diferença no último bit do
`sin`/`cos` da plataforma leva o integrador por uma sequência de passos
diferente, porém igualmente válida. A escala da discordância é a própria
tolerância do integrador, `rtol=1e-7`.

Isso importa para como ler os números. Contagens e taxas de acerto são inteiros
e reproduzem exatamente. Estatísticas contínuas — CEP, desvios padrão dos
erros — devem ser citadas como concordantes em cerca de seis algarismos
significativos, não até o último dígito. O `tests/test_naval_pipeline.py` é o
que atribui a diferença à plataforma, e não à refatoração.

### Rodando as conferências

```bash
pytest                                          # 181 testes, ~3 min
pytest tests/test_regression_vs_original.py -v  # a suíte bit a bit
pytest tests/test_naval_pipeline.py -v          # a pipeline do TCC
pytest -m "not slow"                            # pula as integrações completas
```

---

## O que mudou em relação à versão anterior

A estrutura anterior está na branch [`legacy`](../../tree/legacy) e continua
funcionando. Era um arquivo único mais trechos de exemplo que precisavam ser
**colados no fim daquele arquivo** para rodar. Esta branch mantém a física e
muda o empacotamento:

- **O motor virou pacote.** Dez módulos focados em vez de um arquivo, com a
  parte gráfica separada — um laço de Monte Carlo nunca importa matplotlib.
- **Os exemplos viraram scripts.** `python examples/01_single_shot.py`, com
  argumentos de linha de comando, em vez de copiar e colar.
- **Nenhum caminho fixo.** Os antigos `C:\Users\DELL\Downloads\...` sumiram; os
  arquivos de dados são resolvidos em relação ao repositório.
- **O apêndice antiaéreo virou módulo**, e genérico: qualquer alvo por faces,
  qualquer distribuição polar de fragmentos.
- **A campanha do TCC é reprodutível.** Varredura, seleção, dispersão e custo
  são código de biblioteca com um script por etapa, e as tabelas intermediárias
  publicadas estão em `data/`.
- **Suíte de testes e provas executáveis** — 181 testes mais os scripts em
  `scripts/`, que sustentam as afirmações de equivalência.
- **Coeficientes fornecidos direto** — o `AerodynamicCoefficients` recebe os
  sete que a equação lê, na definição do McCoy, em vez de catorze colunas das
  quais quatro são mortas. Converter uma tabulação de origem ficou de propósito
  fora do pacote: o `examples/07_bring_your_own_table.py` é o caso trabalhado, e
  o `docs/table_5in38_provenance.md` declara os dois pontos em que a tabela do
  5"/38 que vem junto se afasta desse contrato.
- **Um bug real corrigido.** O original convertia o resultado do interpolador
  com `float(x)` sobre um array 1×1, o que o NumPy 2.0 rejeita — ou seja, o
  `Motor.py` não roda mais em uma pilha SciPy/NumPy atual. A correção pega o
  primeiro elemento, que é exatamente o que o `float()` fazia, então nenhum
  número muda.
- **Identificadores e docstrings em inglês**, seguindo os nomes de classe que o
  original já usava, para leitura internacional. As mensagens de console
  continuam em português, sem alteração.

---

## Procedência

Quem escreveu e quem conferiu o quê, na ordem:

**O motor original (branch `legacy`).** Escrito por Luiz Guilherme de Padua
Sanches com auxílio de modelos de linguagem (Claude e ChatGPT), **revisado e
verificado pelo autor**, e usado no seu Trabalho de Conclusão de Curso em
Matemática Aplicada e Computacional no IME-USP. Todo gráfico e toda tabela do
trabalho escrito saíram desse código. É o artefato verificado.

**Esta refatoração (branch `main`).** Escrita pelo **Claude** (assistente de IA
da Anthropic) a pedido do autor, numa única sessão de trabalho, junto com a
suíte de testes e a prova de equivalência. O autor definiu os requisitos,
escolheu as decisões de projeto e revisou e aprovou o resultado — mas os números
do TCC não foram refeitos a partir desta branch, e ela não fez parte do trabalho
entregue à universidade. É novidade.

O que isso significa na prática: a equivalência com a `legacy` é verificada por
máquina e reproduzível por qualquer um (`scripts/proof_of_equivalence.py`),
então os números não estão em questão. O que não houve foi uma segunda revisão
humana do código reestruturado em si. Trate a `legacy` como o artefato citável e
a `main` como o mantido.

**Fontes dos dados físicos.** Os coeficientes aerodinâmicos e a distribuição de
fragmentos vêm de relatórios de artilharia publicados; as fontes específicas
estão citadas nos módulos que as usam (`sixdof/aa/presets.py` para a ogiva,
`docs/shahed_target_geometry.pt-BR.md` para a geometria do alvo).

## Citação

Para uso acadêmico, cite o TCC junto com o software — veja
[`CITATION.cff`](CITATION.cff). Diga qual branch você usou.

## Licença

MIT — veja [`LICENSE`](LICENSE). Você pode usar, modificar e redistribuir
livremente, inclusive comercialmente, desde que o aviso de copyright e o texto
da licença acompanhem o código. Para uso acadêmico, cite também o TCC.

## Referência

McCoy, R. L. *Modern Exterior Ballistics: The Launch and Flight Dynamics of
Symmetric Projectiles.* 2ª ed. Schiffer Publishing, 2012.
