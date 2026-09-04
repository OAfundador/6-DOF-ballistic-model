# Geometria Simplificada do Alvo Shahed

Este documento descreve apenas a geometria do alvo. Ela entra no modelo para
calcular a area projetada do alvo vista pela nuvem de fragmentos.

No codigo, o preset esta em `src/sixdof/aa/presets.py` (`shahed_136`), construido
pelo builder generico `triangular_prism_target` de `src/sixdof/aa/geometry.py`.
A projecao descrita abaixo e o metodo `Target.projected_area`. As areas e normais
deste documento sao verificadas em `tests/test_aa_damage.py`.

## Convencao de Eixos

O alvo esta nivelado e com orientacao fixa:

```text
x: direcao do nariz
y: vertical
z: lateral / envergadura
```

A direcao usada no calculo e:

```text
d = direcao_chegada = vetor unitario burst -> alvo
```

Ou seja, `d` aponta na direcao em que os fragmentos viajam ate o centro do alvo.

## Dimensoes

```text
comprimento  = 3.5 m
envergadura  = 2.5 m
espessura    = 0.35 m
meia asa     = envergadura / 2 = 1.25 m
```

A forma e um prisma triangular, tipo "bloco de queijo":

```text
vista superior aproximada:

        nariz (+x)
           /\
          /  \
         /    \
        /______\
       cauda / traseira
```

O triangulo de planta e extrudado na espessura vertical `y`.

## Faces do Modelo

Nao existe uma face frontal plana no nariz. O nariz e tratado como uma aresta.
As faces com area finita sao:

```text
superior
inferior
lateral_direita
lateral_esquerda
traseira
```

Areas:

```text
area_superior = 0.5 * envergadura * comprimento
              = 0.5 * 2.5 * 3.5
              = 4.375 m2

area_inferior = area_superior
              = 4.375 m2

comprimento_lateral = sqrt(comprimento^2 + (envergadura/2)^2)
                    = sqrt(3.5^2 + 1.25^2)
                    = 3.7165 m

area_lateral = comprimento_lateral * espessura
             = 3.7165 * 0.35
             = 1.3008 m2 por lateral

area_traseira = envergadura * espessura
              = 2.5 * 0.35
              = 0.875 m2

volume_aproximado = area_superior * espessura
                  = 4.375 * 0.35
                  = 1.53125 m3
```

## Normais das Faces

As normais sao vetores unitarios apontando para fora do prisma.

```text
superior: (0,  1, 0)
inferior: (0, -1, 0)
traseira: (-1, 0, 0)
```

Para as laterais inclinadas:

```text
Llat = sqrt(comprimento^2 + meia_asa^2)

lateral_direita:
    n = (meia_asa / Llat, 0, comprimento / Llat)

lateral_esquerda:
    n = (meia_asa / Llat, 0, -comprimento / Llat)
```

Numericamente:

```text
meia_asa / Llat      = 1.25 / 3.7165 = 0.3363
comprimento / Llat   = 3.5  / 3.7165 = 0.9417

lateral_direita  ~= (0.3363, 0,  0.9417)
lateral_esquerda ~= (0.3363, 0, -0.9417)
```

## Escolha da Area Visivel

Para cada face, o codigo calcula:

```text
visibilidade_face = max(0, - dot(normal_saida, d))
contribuicao_face = area_face * visibilidade_face
```

A area exposta total e:

```text
area_exposta = soma(contribuicao_face)
```

Interpretacao:

- Se `dot(normal_saida, d) < 0`, a face esta voltada para a origem dos fragmentos e contribui.
- Se `dot(normal_saida, d) > 0`, a face esta no lado oposto e nao contribui.
- Se `dot(normal_saida, d) = 0`, a face esta de lado para a nuvem e contribui zero.

Essa regra evita somar faces opostas ao mesmo tempo. Por exemplo, superior e inferior nao aparecem simultaneamente para uma chegada puramente vertical.

## Exemplos Simples

Nuvem vindo de cima:

```text
d = (0, -1, 0)
area visivel = superior = 4.375 m2
```

Nuvem vindo de baixo:

```text
d = (0, 1, 0)
area visivel = inferior = 4.375 m2
```

Nuvem vindo da cauda:

```text
d = (1, 0, 0)
area visivel = traseira = 0.875 m2
```

Nuvem vindo de um lado puro:

```text
d = (0, 0, -1)
area visivel = projecao da lateral_direita
             = area_lateral * (comprimento / Llat)
             = (Llat * espessura) * (comprimento / Llat)
             = comprimento * espessura
             = 3.5 * 0.35
             = 1.225 m2
```

Nuvem vindo do nariz:

```text
d = (-1, 0, 0)
area visivel = projecao das duas laterais inclinadas
             = 2 * area_lateral * (meia_asa / Llat)
             = 2 * meia_asa * espessura
             = envergadura * espessura
             = 0.875 m2
```

## Observacao Importante

Esta e uma area projetada equivalente de um corpo convexo simplificado. O codigo nao faz intersecao geometrica detalhada raio-malha e nao modela componentes internos do drone. A area resultante alimenta a etapa de fragmentos esperados:

```text
fragmentos_esperados = densidade_fragmentos * area_exposta
```

