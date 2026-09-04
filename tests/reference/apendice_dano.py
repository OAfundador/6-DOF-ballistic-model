"""
Apendice simples: alvo Shahed, fuze de proximidade e dano por fragmentos.

Este arquivo existe para verificacao de artigo. A ideia e ser legivel e direto,
parecido com o estilo do codigo legado: parametros no topo, funcoes pequenas e
um unico ponto de entrada chamado pela main do legacy_motor.py.
"""

import numpy as np
from math import acos, atan2, cos, exp, inf, pi, radians, sin, sqrt


# =============================================================================
# PARAMETROS DO ALVO
# =============================================================================

def criar_alvo_shahed(centro=(16673.0, 200.0, 0.7)):
    comprimento = 3.5
    envergadura = 2.5
    espessura = 0.35

    # Modelo "bloco de queijo": prisma triangular extrudado na espessura.
    # Superior/inferior: triangulos de planta. Laterais: retangulos inclinados.
    # Traseira: retangulo na base do triangulo.
    meia_envergadura = 0.5 * envergadura
    comprimento_lateral = sqrt(comprimento**2 + meia_envergadura**2)

    area_superior = 0.5 * envergadura * comprimento
    area_inferior = area_superior
    area_lateral = comprimento_lateral * espessura
    area_traseira = envergadura * espessura
    volume = area_superior * espessura
    normal_lateral_direita = np.array(
        [meia_envergadura / comprimento_lateral, 0.0, comprimento / comprimento_lateral],
        dtype=float,
    )
    normal_lateral_esquerda = np.array(
        [meia_envergadura / comprimento_lateral, 0.0, -comprimento / comprimento_lateral],
        dtype=float,
    )

    faces_projetadas = [
        ("superior", area_superior, np.array([0.0, 1.0, 0.0], dtype=float)),
        ("inferior", area_inferior, np.array([0.0, -1.0, 0.0], dtype=float)),
        ("lateral_direita", area_lateral, normal_lateral_direita),
        ("lateral_esquerda", area_lateral, normal_lateral_esquerda),
        ("traseira", area_traseira, np.array([-1.0, 0.0, 0.0], dtype=float)),
    ]

    return {
        "nome": "Shahed-136 / Geran-2",
        "modelo_geometrico": "prisma triangular simplificado",
        "centro": np.array(centro, dtype=float),
        # Hipotese atual: alvo nivelado, parado e com orientacao fixa.
        # x = nariz/cauda, y = vertical, z = lateral.
        "direcao_nariz": np.array([1.0, 0.0, 0.0], dtype=float),
        "eixo_vertical": np.array([0.0, 1.0, 0.0], dtype=float),
        "eixo_lateral": np.array([0.0, 0.0, 1.0], dtype=float),
        "normal": np.array([0.0, 1.0, 0.0], dtype=float),  # mantido como alias
        "comprimento_m": comprimento,
        "envergadura_m": envergadura,
        "espessura_m": espessura,
        "comprimento_lateral_m": comprimento_lateral,
        "area_superior_m2": area_superior,
        "area_inferior_m2": area_inferior,
        "area_lateral_m2": area_lateral,
        "area_laterais_total_m2": 2.0 * area_lateral,
        "area_traseira_m2": area_traseira,
        "volume_aproximado_m3": volume,
        "faces_projetadas": faces_projetadas,
    }


def vetor_unitario(v):
    v = np.array(v, dtype=float)
    norma = float(np.linalg.norm(v))
    if norma <= 1e-12:
        raise ValueError("vetor com norma nula")
    return v / norma


def area_exposta_shahed(alvo, direcao_chegada):
    """
    Area projetada do alvo vista pela nuvem de fragmentos.

    A direcao_chegada e o vetor burst -> alvo. A orientacao do alvo ainda e
    fixa por hipotese, mas ja fica escrita por eixos para permitir rotacao
    posterior sem mudar a formula.

    Esta area projetada e a adaptacao 3D usada para a Equacao 10 do artigo:
    em vez de usar S*sin(phi) em um unico plano, projetamos as faces
    equivalentes do prisma triangular simplificado na direcao da nuvem.
    """
    d = vetor_unitario(direcao_chegada)
    contribuicoes = {}
    for nome, area, normal_saida in alvo["faces_projetadas"]:
        normal_saida = vetor_unitario(normal_saida)
        visibilidade = max(0.0, -float(np.dot(d, normal_saida)))
        contribuicoes[nome] = area * visibilidade

    return sum(contribuicoes.values()), contribuicoes


def obliquidade_efetiva_shahed(alvo, direcao_chegada, contribuicoes):
    """
    Obliquidade aproximada usando a face que mais contribui para a area vista.
    """
    d = vetor_unitario(direcao_chegada)
    face = max(contribuicoes, key=contribuicoes.get)
    normais = {nome: vetor_unitario(normal) for nome, _, normal in alvo["faces_projetadas"]}
    cos_face = max(0.0, -float(np.clip(np.dot(d, normais[face]), -1.0, 1.0)))
    return acos(cos_face), face


# =============================================================================
# PARAMETROS DA OGIVA
# =============================================================================

def criar_ogiva_vt_fcl_mk49():
    # Fonte principal: NPG Report No. 1124, PDF pp. 5, 8-10 e 22.
    # A versao atual usa o total efetivo diretamente.
    n_fragmentos_efetivos = 2113

    return {
        "nome": '5"/38 VT(FCL) Mk 49 Comp A-3',
        # 4080 ft/s convertido para m/s. O relatório chama isso de velocidade
        # mediana dos fragmentos na zona 80-110 graus, medida nos primeiros 30 ft.
        # Aqui ela é usada como aproximação simples para v0 do modelo de dano.
        "v0_m_s": 1243.6,
        # Distribuicao polar experimental de hits, medida em relacao ao eixo
        # longitudinal do projetil. Fonte: NPG 1124, PDF pp. 9-10.
        "zonas_polares_hits": [
            (0.0, 15.0, 10),
            (15.0, 40.0, 0),
            (40.0, 65.0, 30),
            (65.0, 115.0, 1072),
            (115.0, 165.0, 95),
            (165.0, 180.0, 10),
        ],
        "n_fragmentos_efetivos": n_fragmentos_efetivos,
    }


# =============================================================================
# FUZE DE PROXIMIDADE
# =============================================================================

def avaliar_fuze_proximidade(resultado, alvo, raio_m=24.38, tempo_armar_s=0.5):
    centro = alvo["centro"]
    melhor_idx = None
    melhor_dist = inf

    if getattr(resultado, "stop_reason", None) == "fuze":
        idx = len(resultado.t) - 1
        pos = np.array([resultado.x[idx], resultado.y[idx], resultado.z[idx]], dtype=float)
        return {
            "acionado": True,
            "origem": "evento_integrador",
            "idx": idx,
            "tempo_s": float(resultado.t[idx]),
            "posicao_m": pos,
            "velocidade_m_s": np.array(
                [resultado.V1[idx], resultado.V2[idx], resultado.V3[idx]], dtype=float
            ),
            "eixo_projetil_i": np.array(
                [resultado.i1[idx], resultado.i2[idx], resultado.i3[idx]], dtype=float
            ),
            "distancia_m": float(np.linalg.norm(pos - centro)),
        }

    for i in range(len(resultado.t)):
        if resultado.t[i] < tempo_armar_s:
            continue

        pos = np.array([resultado.x[i], resultado.y[i], resultado.z[i]], dtype=float)
        dist = float(np.linalg.norm(pos - centro))

        if dist < melhor_dist:
            melhor_idx = i
            melhor_dist = dist

        if dist <= raio_m + 1e-6:
            return {
                "acionado": True,
                "origem": "amostra_trajetoria",
                "idx": i,
                "tempo_s": float(resultado.t[i]),
                "posicao_m": pos,
                "velocidade_m_s": np.array(
                    [resultado.V1[i], resultado.V2[i], resultado.V3[i]], dtype=float
                ),
                "eixo_projetil_i": np.array(
                    [resultado.i1[i], resultado.i2[i], resultado.i3[i]], dtype=float
                ),
                "distancia_m": dist,
            }

    if melhor_idx is None:
        return None

    return {
        "acionado": False,
        "origem": "menor_distancia_amostrada",
        "idx": melhor_idx,
        "tempo_s": float(resultado.t[melhor_idx]),
        "posicao_m": np.array(
            [resultado.x[melhor_idx], resultado.y[melhor_idx], resultado.z[melhor_idx]],
            dtype=float,
        ),
        "velocidade_m_s": np.array(
            [resultado.V1[melhor_idx], resultado.V2[melhor_idx], resultado.V3[melhor_idx]],
            dtype=float,
        ),
        "eixo_projetil_i": np.array(
            [resultado.i1[melhor_idx], resultado.i2[melhor_idx], resultado.i3[melhor_idx]],
            dtype=float,
        ),
        "distancia_m": melhor_dist,
    }


# =============================================================================
# MODELO DE DANO
# =============================================================================

def velocidade_fragmento_dinamica(ogiva, velocidade_projetil_m_s, phi_rad):
    v0 = ogiva["v0_m_s"]
    v1 = velocidade_projetil_m_s
    return sqrt(v1**2 + v0**2 + 2.0 * v1 * v0 * cos(phi_rad))


def alpha2_de_alpha1(ogiva, velocidade_projetil_m_s, alpha1_rad):
    v0 = ogiva["v0_m_s"]
    v1 = velocidade_projetil_m_s
    return atan2(v0 * sin(alpha1_rad), v0 * cos(alpha1_rad) + v1)


def angulo_entre_direcoes(direcao_a, direcao_b):
    cos_angulo = float(np.clip(np.dot(direcao_a, direcao_b), -1.0, 1.0))
    return acos(cos_angulo)


def phi_dispersao_de_eixo(direcao_eixo, direcao_fragmento):
    """
    Angulo phi do artigo entre a direcao de dispersao e o eixo da ogiva.

    Para o nosso alvo pontual, a direcao de dispersao observada e burst->alvo.
    Este phi nao e alpha1: alpha1 e o angulo estatico das zonas polares da
    ogiva, e alpha2 e calculado a partir de alpha1 pela equacao do artigo.
    """
    return angulo_entre_direcoes(direcao_eixo, direcao_fragmento)


def densidade_angular_npg(
    ogiva,
    n_fragmentos,
    distancia_m,
    phi_alvo_rad,
    velocidade_projetil_m_s,
):
    """
    Densidade analitica local baseada nas zonas polares do NPG com correcao dinamica.

    As zonas do NPG sao intervalos estaticos alpha1. Cada limite alpha1 e
    convertido para alpha2 usando a formula do artigo. O alvo entra pelo phi,
    isto e, pelo angulo entre eixo i e a direcao burst->alvo.

    Como a versao simplificada assume penetracao total, n_fragmentos e o numero
    efetivo total da ogiva, distribuido nas zonas pela proporcao de hits NPG.
    """
    if distancia_m <= 0.0 or n_fragmentos <= 0.0:
        return 0.0, None, None, 0.0, 0.0

    zonas_dinamicas = []
    for theta_min_deg, theta_max_deg, hits in ogiva["zonas_polares_hits"]:
        alpha2_min = alpha2_de_alpha1(ogiva, velocidade_projetil_m_s, radians(theta_min_deg))
        alpha2_max = alpha2_de_alpha1(ogiva, velocidade_projetil_m_s, radians(theta_max_deg))
        if alpha2_min > alpha2_max:
            alpha2_min, alpha2_max = alpha2_max, alpha2_min
        zonas_dinamicas.append((theta_min_deg, theta_max_deg, alpha2_min, alpha2_max, hits))

    zona = None
    for idx, item in enumerate(zonas_dinamicas):
        theta_min_deg, theta_max_deg, alpha2_min, alpha2_max, hits = item
        ultima = idx == len(zonas_dinamicas) - 1
        if alpha2_min <= phi_alvo_rad < alpha2_max or (
            ultima and alpha2_min <= phi_alvo_rad <= alpha2_max
        ):
            zona = item
            break

    if zona is None:
        return 0.0, None, None, 0.0, 0.0

    theta_min_deg, theta_max_deg, alpha2_min, alpha2_max, hits_zona = zona
    total_hits = sum(hits for _, _, hits in ogiva["zonas_polares_hits"])
    n_zona = n_fragmentos * hits_zona / total_hits

    area_faixa = 2.0 * pi * distancia_m**2 * abs(cos(alpha2_min) - cos(alpha2_max))
    if area_faixa <= 0.0:
        return 0.0, None, None, 0.0, 0.0

    zona_estatica = (theta_min_deg, theta_max_deg, hits_zona)
    zona_dinamica = (float(np.degrees(alpha2_min)), float(np.degrees(alpha2_max)))
    return n_zona / area_faixa, zona_estatica, zona_dinamica, n_zona, area_faixa


def probabilidade_destruicao_bernoulli(fragmentos_esperados):
    """
    Probabilidade de pelo menos um fragmento cruzar o alvo.

    Hipoteses desta versao:
    1. todo fragmento que cruza a area projetada penetra;
    2. qualquer penetracao e dano critico/destruicao do alvo.

    Com M fragmentos esperados na area, modelamos a contagem por Poisson e
    obtemos uma Bernoulli de destruicao: p = 1 - exp(-M).
    """
    if fragmentos_esperados <= 0.0:
        return 0.0
    return float(np.clip(1.0 - exp(-fragmentos_esperados), 0.0, 1.0))


def avaliar_dano_fragmentario(
    ponto_burst,
    velocidade_projetil,
    eixo_projetil_i,
    alvo,
    ogiva,
    fuze_acionado=True,
):
    vetor_alvo = alvo["centro"] - ponto_burst
    distancia_m = float(np.linalg.norm(vetor_alvo))
    if distancia_m <= 1e-6:
        raise ValueError("burst coincidente com o centro do alvo")

    direcao_fragmento = vetor_alvo / distancia_m
    velocidade_proj = float(np.linalg.norm(velocidade_projetil))
    direcao_velocidade = velocidade_projetil / velocidade_proj

    eixo_norma = float(np.linalg.norm(eixo_projetil_i))
    if eixo_norma <= 1e-12:
        raise ValueError("eixo i do projetil tem norma nula")
    direcao_eixo = eixo_projetil_i / eixo_norma

    phi_alvo = phi_dispersao_de_eixo(direcao_eixo, direcao_fragmento)
    phi_velocidade = phi_alvo

    angulo_vel_frag = angulo_entre_direcoes(direcao_velocidade, direcao_fragmento)

    area_exposta, contribuicoes_area = area_exposta_shahed(alvo, direcao_fragmento)
    obliquidade, face_dominante = obliquidade_efetiva_shahed(
        alvo, direcao_fragmento, contribuicoes_area
    )

    v_frag = velocidade_fragmento_dinamica(ogiva, velocidade_proj, phi_velocidade)

    n_fragmentos_modelo = ogiva["n_fragmentos_efetivos"]
    densidade, zona, zona_dinamica, n_zona, area_faixa = densidade_angular_npg(
        ogiva, n_fragmentos_modelo, distancia_m, phi_alvo, velocidade_proj
    )
    if zona is None:
        alpha1_zona_centro = np.nan
    else:
        alpha1_zona_centro = 0.5 * (zona[0] + zona[1])

    fragmentos_esperados = max(0.0, densidade * area_exposta)
    p_total = probabilidade_destruicao_bernoulli(fragmentos_esperados)

    return {
        "fuze_acionado": fuze_acionado,
        "distancia_m": distancia_m,
        "phi_alvo_deg": np.degrees(phi_alvo),
        "alpha1_zona_centro_deg": alpha1_zona_centro,
        "phi_velocidade_deg": np.degrees(phi_velocidade),
        "angulo_vel_fragmento_deg": np.degrees(angulo_vel_frag),
        "zona_polar_npg": zona,
        "zona_polar_dinamica_deg": zona_dinamica,
        "obliquidade_deg": np.degrees(obliquidade),
        "face_alvo_dominante": face_dominante,
        "contribuicoes_area_m2": contribuicoes_area,
        "area_exposta_m2": area_exposta,
        "v_frag_m_s": v_frag,
        "n_fragmentos_modelo": n_fragmentos_modelo,
        "n_fragmentos_zona": n_zona,
        "area_faixa_dinamica_m2": area_faixa,
        "densidade_efetiva": densidade,
        "fragmentos_esperados_area": fragmentos_esperados,
        "fragmentos_no_alvo": fragmentos_esperados,
        "modelo_probabilistico": "Bernoulli/Poisson: p=1-exp(-M)",
        "hipotese_penetracao_total": True,
        "hipotese_dano_critico_por_fragmento": True,
        "p_destruicao": p_total,
        "p_dano_total": p_total,
    }


def avaliar_fuze_e_dano(resultado, alvo, ogiva, raio_fuze_m=24.38, tempo_armar_s=0.5):
    burst = avaliar_fuze_proximidade(resultado, alvo, raio_fuze_m, tempo_armar_s)
    if burst is None:
        return None, None

    dano = avaliar_dano_fragmentario(
        ponto_burst=burst["posicao_m"],
        velocidade_projetil=burst["velocidade_m_s"],
        eixo_projetil_i=burst["eixo_projetil_i"],
        alvo=alvo,
        ogiva=ogiva,
        fuze_acionado=burst["acionado"],
    )
    return burst, dano
