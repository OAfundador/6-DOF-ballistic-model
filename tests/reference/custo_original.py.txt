Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
"""
CÁLCULO DO VALOR ESPERADO PARA ANÁLISE DE CUSTO DE ENGAJAMENTO

Fórmula:
E[Custo] = c(1 + (1-p1) + (1-p1)(1-p2) + ... + (1-p1)...(1-pn-1)) + (1-p1)...(1-pn)C

... Onde:
... - c: custo de cada munição
... - C: custo/valor da embarcação (penalidade por falha total)
... - pi: probabilidade de acerto do projétil i
... """
... 
... import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... 
... # ============================================================================
... # CONFIGURAÇÕES - AJUSTE AQUI OS PARÂMETROS
... # ============================================================================
... 
... # Arquivo de entrada com as taxas de acerto
... ARQUIVO_ENTRADA = r'C:\Users\DELL\AppData\Local\Programs\Microsoft VS Code\segundomonte_carlo_massivo_alpha0_beta0_elev01_azi005.xlsx'
... 
... # Parâmetros de custo (em USD - 12/11/2025)
... CUSTO_MUNICAO = 2000        # c: Custo por munição
... CUSTO_EMBARCACAO = 289000000  # C: Valor da embarcação/penalidade
... 
... # ============================================================================
... # FUNÇÕES
... # ============================================================================
... 
... def calcular_valor_esperado(probs, custo_municao, custo_embarcacao):
...     """
...     Calcula o valor esperado usando a fórmula:
...     E[Custo] = c(1 + (1-p1) + (1-p1)(1-p2) + ... + (1-p1)...(1-pn-1)) + (1-p1)...(1-pn)C
...     
...     Parâmetros:
...     -----------
...     probs : array
...         Array com as probabilidades de acerto de cada disparo
...     custo_municao : float
...         Custo de cada munição (c)
...     custo_embarcacao : float
...         Custo da embarcação/penalidade (C)
    
    Retorna:
    --------
    dict com os resultados
    """
    n = len(probs)
    
    # TERMO 1: Custo esperado de munição
    # c × [1 + (1-p1) + (1-p1)(1-p2) + ... + (1-p1)(1-p2)...(1-pn-1)]
    termo1_fatores = []
    prob_acumulada_falha = 1.0
    
    for i in range(n):
        termo1_fatores.append(prob_acumulada_falha)
        if i < n - 1:  # Não multiplica no último
            prob_acumulada_falha *= (1 - probs[i])
    
    num_esperado_disparos = sum(termo1_fatores)
    termo1 = custo_municao * num_esperado_disparos
    
    # TERMO 2: Custo da falha total
    # C × (1-p1)(1-p2)...(1-pn)
    prob_falha_total = np.prod([1 - p for p in probs])
    termo2 = custo_embarcacao * prob_falha_total
    
    # Custo total esperado
    custo_total = termo1 + termo2
    
    # Probabilidade de sucesso (destruir o alvo)
    prob_sucesso = 1 - prob_falha_total
    
    return {
        'n_disparos': n,
        'num_esperado_disparos': num_esperado_disparos,
        'termo1_custo_municao': termo1,
        'prob_falha_total': prob_falha_total,
        'termo2_custo_falha': termo2,
        'custo_total_esperado': custo_total,
        'prob_sucesso': prob_sucesso
    }

# ============================================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================================

print("=" * 80)
print("CÁLCULO DO VALOR ESPERADO - ANÁLISE DE CUSTO DE ENGAJAMENTO")
print("=" * 80)

# Ler o arquivo Excel com as taxas de acerto
print(f"\nLendo arquivo: {ARQUIVO_ENTRADA}")
df = pd.read_excel(ARQUIVO_ENTRADA, sheet_name=0)

# Converter taxa de acerto (%) para probabilidade
df['Prob_acerto'] = df['Taxa_acerto_Drone_Sea_Baby_pct'] / 100
probabilidades = df['Prob_acerto'].values
n_total = len(probabilidades)

print(f"\n{'='*80}")
print(f"DADOS DE ENTRADA")
print(f"{'='*80}")
print(f"Número total de disparos: {n_total}")
print(f"Taxa de acerto:")
print(f"  Mínima:  {df['Taxa_acerto_Drone_Sea_Baby_pct'].min():.4f}%")
print(f"  Média:   {df['Taxa_acerto_Drone_Sea_Baby_pct'].mean():.4f}%")
print(f"  Máxima:  {df['Taxa_acerto_Drone_Sea_Baby_pct'].max():.4f}%")
print(f"  Mediana: {df['Taxa_acerto_Drone_Sea_Baby_pct'].median():.4f}%")

print(f"\n{'='*80}")
print(f"PARÂMETROS DE CUSTO")
print(f"{'='*80}")
print(f"Custo por munição (c):      USD {CUSTO_MUNICAO:>15,.2f}")
print(f"Valor da embarcação (C):    USD {CUSTO_EMBARCACAO:>15,.2f}")
print(f"Razão C/c:                  {CUSTO_EMBARCACAO/CUSTO_MUNICAO:>15,.1f}x")

# Calcular com TODOS os disparos
print(f"\n{'='*80}")
print(f"RESULTADO COM TODOS OS {n_total} DISPAROS")
print(f"{'='*80}")

resultado = calcular_valor_esperado(probabilidades, CUSTO_MUNICAO, CUSTO_EMBARCACAO)

print(f"\n📊 TERMO 1 - Custo Esperado de Munição:")
print(f"{'─'*80}")
print(f"   Número esperado de disparos:  {resultado['num_esperado_disparos']:>12.4f}")
print(f"   Custo esperado de munição:    USD {resultado['termo1_custo_municao']:>15,.2f}")

print(f"\n📊 TERMO 2 - Custo de Falha Total:")
print(f"{'─'*80}")
print(f"   Probabilidade de falha total: {resultado['prob_falha_total']:>12.10f}")
print(f"                                 {resultado['prob_falha_total']*100:>12.8f}%")
print(f"   Custo esperado de falha:      USD {resultado['termo2_custo_falha']:>15,.2f}")

print(f"\n🎯 RESULTADO FINAL:")
print(f"{'═'*80}")
print(f"   CUSTO ESPERADO TOTAL:         USD {resultado['custo_total_esperado']:>15,.2f}")
print(f"   PROBABILIDADE DE SUCESSO:     {resultado['prob_sucesso']:>12.10f}")
print(f"                                 {resultado['prob_sucesso']*100:>12.8f}%")
print(f"{'═'*80}")

# Análise incremental detalhada
print(f"\n{'='*80}")
print(f"ANÁLISE INCREMENTAL DETALHADA")
print(f"{'='*80}")

pontos_analise = [1, 2, 3, 5, 10, 20, 30, 40, 50, 75, 100, 130, n_total]
print(f"\n{'N':>5} | {'P(Sucesso)%':>14} | {'P(Falha)%':>14} | {'E[N]':>10} | {'E[Custo] (USD)':>18}")
print(f"{'─'*80}")

for n in pontos_analise:
    if n > n_total:
        continue
    res = calcular_valor_esperado(probabilidades[:n], CUSTO_MUNICAO, CUSTO_EMBARCACAO)
    print(f"{n:>5} | {res['prob_sucesso']*100:>13.6f}% | {res['prob_falha_total']*100:>13.6f}% | " +
          f"{res['num_esperado_disparos']:>10.4f} | {res['custo_total_esperado']:>18,.2f}")

# Encontrar ponto de custo mínimo
print(f"\n{'='*80}")
print(f"ANÁLISE DE PONTOS CRÍTICOS")
print(f"{'='*80}")

# Calcular para todos os pontos
todos_resultados = []
for n in range(1, n_total + 1):
    res = calcular_valor_esperado(probabilidades[:n], CUSTO_MUNICAO, CUSTO_EMBARCACAO)
    todos_resultados.append(res)

# Encontrar mínimo custo
custos = [r['custo_total_esperado'] for r in todos_resultados]
idx_min_custo = np.argmin(custos)
n_min_custo = idx_min_custo + 1
resultado_min = todos_resultados[idx_min_custo]

print(f"\n🎯 PONTO DE CUSTO MÍNIMO:")
print(f"{'─'*80}")
print(f"   N ótimo:                      {n_min_custo:>12}")
print(f"   Custo mínimo:                 USD {resultado_min['custo_total_esperado']:>15,.2f}")
print(f"   P(Sucesso):                   {resultado_min['prob_sucesso']*100:>12.6f}%")
print(f"   E[N]:                         {resultado_min['num_esperado_disparos']:>12.4f}")

# Encontrar pontos com P(Sucesso) >= 90%, 95%, 99%
niveis_confianca = [0.90, 0.95, 0.99]
print(f"\n🎯 PONTOS PARA NÍVEIS DE CONFIANÇA:")
print(f"{'─'*80}")

for nivel in niveis_confianca:
    for i, res in enumerate(todos_resultados):
        if res['prob_sucesso'] >= nivel:
            n_nivel = i + 1
            print(f"   P(Sucesso) ≥ {nivel*100:>5.1f}%:")
            print(f"      N mínimo:                  {n_nivel:>12}")
            print(f"      Custo:                     USD {res['custo_total_esperado']:>15,.2f}")
            print(f"      P(Sucesso) real:           {res['prob_sucesso']*100:>12.6f}%")
            print(f"      E[N]:                      {res['num_esperado_disparos']:>12.4f}")
            print()
            break

# Comparação de estratégias
print(f"{'='*80}")
print(f"COMPARAÇÃO DE ESTRATÉGIAS")
print(f"{'='*80}")

estrategias = {
    'Conservadora (1 disparo)': 1,
    'Moderada (N ótimo)': n_min_custo,
    'Agressiva (P≥95%)': None,
    'Total (todos)': n_total
}

# Encontrar N para P≥95%
for i, res in enumerate(todos_resultados):
    if res['prob_sucesso'] >= 0.95:
        estrategias['Agressiva (P≥95%)'] = i + 1
        break

print(f"\n{'Estratégia':<30} | {'N':>5} | {'P(Sucesso)%':>14} | {'E[Custo] USD':>18}")
print(f"{'─'*80}")

for nome, n in estrategias.items():
    if n is None:
        print(f"{nome:<30} | {'N/A':>5} | {'N/A':>14} | {'N/A':>18}")
    else:
        res = todos_resultados[n-1]
        print(f"{nome:<30} | {n:>5} | {res['prob_sucesso']*100:>13.6f}% | {res['custo_total_esperado']:>18,.2f}")

# ============================================================================
# GRÁFICOS
# ============================================================================

print(f"\n{'='*80}")
print(f"GERANDO GRÁFICOS...")
print(f"{'='*80}")

n_disparos_range = range(1, n_total + 1)

# Calcular dados para os gráficos
prob_sucesso_range = [todos_resultados[i]['prob_sucesso'] * 100 for i in range(n_total)]
custo_range = [todos_resultados[i]['custo_total_esperado'] for i in range(n_total)]
custo_municao_range = [todos_resultados[i]['termo1_custo_municao'] for i in range(n_total)]

# ============================================================================
# GRÁFICO 1: PROBABILIDADE DE SUCESSO (PONTOS)
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 8))

# Plotar como PONTOS
ax1.scatter(n_disparos_range, prob_sucesso_range,
           s=30, c='blue', marker='o', alpha=0.6, edgecolors='darkblue',
           linewidths=0.5, label='P(Sucesso)')

# Linhas de referência
ax1.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.6, label='90%')
ax1.axhline(y=95, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='95%')
ax1.axhline(y=99, color='red', linestyle='--', linewidth=2, alpha=0.6, label='99%')

# Marcar ponto ótimo
ax1.scatter(n_min_custo, resultado_min['prob_sucesso']*100,
           s=400, c='red', marker='*', edgecolors='black', linewidths=2,
           zorder=10, label=f'Custo Mínimo (N={n_min_custo})')

ax1.set_xlabel('Número de Disparos', fontsize=14, fontweight='bold')
ax1.set_ylabel('Probabilidade de Sucesso (%)', fontsize=14, fontweight='bold')
ax1.set_title('Probabilidade de Destruir o Alvo vs Número de Disparos',
             fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.legend(fontsize=12, loc='lower right', framealpha=0.95)
ax1.set_ylim([0, 105])
ax1.set_xlim([0, n_total + 1])

# Adicionar anotações
ax1.annotate(f'N={n_min_custo}\nP={resultado_min["prob_sucesso"]*100:.2f}%',
            xy=(n_min_custo, resultado_min['prob_sucesso']*100),
            xytext=(n_min_custo + 10, resultado_min['prob_sucesso']*100 - 10),
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

plt.tight_layout()
plt.savefig('probabilidade_sucesso.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfico 1 salvo: probabilidade_sucesso.png")

# ============================================================================
# GRÁFICO 2: CUSTO ESPERADO TOTAL (PONTOS)
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Plotar como PONTOS
ax2.scatter(n_disparos_range, custo_range,
           s=30, c='red', marker='o', alpha=0.6, edgecolors='darkred',
           linewidths=0.5, label='Custo Esperado Total')

# Marcar ponto de custo mínimo
ax2.scatter(n_min_custo, resultado_min['custo_total_esperado'],
           s=400, c='green', marker='*', edgecolors='black', linewidths=2,
           zorder=10, label=f'Custo Mínimo (N={n_min_custo})')

ax2.set_xlabel('Número de Disparos', fontsize=14, fontweight='bold')
ax2.set_ylabel('Custo Esperado (USD)', fontsize=14, fontweight='bold')
ax2.set_title('Custo Esperado Total vs Número de Disparos',
             fontsize=16, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.legend(fontsize=12, loc='best', framealpha=0.95)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
ax2.set_xlim([0, n_total + 1])

# Adicionar anotações
ax2.annotate(f'CUSTO MÍNIMO\nN={n_min_custo}\n${resultado_min["custo_total_esperado"]/1e6:.3f}M',
            xy=(n_min_custo, resultado_min['custo_total_esperado']),
            xytext=(n_min_custo + 15, resultado_min['custo_total_esperado'] * 1.1),
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

plt.tight_layout()
plt.savefig('custo_esperado.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfico 2 salvo: custo_esperado.png")

# ============================================================================
# GRÁFICO 3: CUSTO DE MUNIÇÃO APENAS (PONTOS)
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))

# Plotar como PONTOS
ax3.scatter(n_disparos_range, custo_municao_range,
           s=30, c='blue', marker='o', alpha=0.6, edgecolors='darkblue',
           linewidths=0.5, label='Custo de Munição')

# Marcar ponto de custo mínimo TOTAL (para referência)
ax3.scatter(n_min_custo, todos_resultados[idx_min_custo]['termo1_custo_municao'],
           s=400, c='orange', marker='*', edgecolors='black', linewidths=2,
           zorder=10, label=f'No Custo Mínimo Total (N={n_min_custo})')

ax3.set_xlabel('Número de Disparos', fontsize=14, fontweight='bold')
ax3.set_ylabel('Custo de Munição (USD)', fontsize=14, fontweight='bold')
ax3.set_title('Custo Esperado de Munição vs Número de Disparos\n(Termo 1 da Fórmula)',
             fontsize=16, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax3.legend(fontsize=12, loc='best', framealpha=0.95)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.1f}k'))
ax3.set_xlim([0, n_total + 1])

# Adicionar anotações
custo_municao_otimo = todos_resultados[idx_min_custo]['termo1_custo_municao']
ax3.annotate(f'No ponto ótimo\nN={n_min_custo}\n${custo_municao_otimo/1e3:.1f}k',
            xy=(n_min_custo, custo_municao_otimo),
            xytext=(n_min_custo + 15, custo_municao_otimo * 1.15),
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

plt.tight_layout()
plt.savefig('custo_municao.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfico 3 salvo: custo_municao.png")

plt.show()

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================

print(f"\n{'='*80}")
print(f"SALVANDO RESULTADOS...")
print(f"{'='*80}")

# Criar DataFrame com análise incremental completa
dados_incrementais = []
for n in range(1, n_total + 1):
    res = todos_resultados[n-1]
    dados_incrementais.append({
        'N_Disparos': n,
        'Prob_Sucesso_%': res['prob_sucesso'] * 100,
        'Prob_Falha_Total_%': res['prob_falha_total'] * 100,
        'Num_Esperado_Disparos': res['num_esperado_disparos'],
        'Custo_Municao_USD': res['termo1_custo_municao'],
        'Custo_Falha_USD': res['termo2_custo_falha'],
        'Custo_Total_Esperado_USD': res['custo_total_esperado']
    })

df_resultados = pd.DataFrame(dados_incrementais)

# Adicionar informações dos pontos de disparo
df_completo = pd.concat([df, df_resultados], axis=1)

# Salvar em Excel com múltiplas abas
with pd.ExcelWriter('resultados_valor_esperado.xlsx', engine='openpyxl') as writer:
    # Aba 1: Análise incremental
    df_resultados.to_excel(writer, sheet_name='Analise_Incremental', index=False)
    
    # Aba 2: Dados completos
    df_completo.to_excel(writer, sheet_name='Dados_Completos', index=False)
    
    # Aba 3: Resumo
    resumo = pd.DataFrame({
        'Parametro': [
            'Numero Total de Disparos',
            'Custo por Municao (USD)',
            'Valor da Embarcacao (USD)',
            'Razao C/c',
            '---TODOS OS DISPAROS---',
            'Probabilidade de Sucesso (%)',
            'Probabilidade de Falha Total (%)',
            'Numero Esperado de Disparos',
            'Custo Esperado de Municao (USD)',
            'Custo Esperado de Falha (USD)',
            'CUSTO TOTAL ESPERADO (USD)',
            '---PONTO OTIMO---',
            'N Otimo (Custo Minimo)',
            'Custo Minimo (USD)',
            'P(Sucesso) no Otimo (%)',
            'E[N] no Otimo',
        ],
        'Valor': [
            n_total,
            CUSTO_MUNICAO,
            CUSTO_EMBARCACAO,
            CUSTO_EMBARCACAO/CUSTO_MUNICAO,
            '',
            resultado['prob_sucesso'] * 100,
            resultado['prob_falha_total'] * 100,
            resultado['num_esperado_disparos'],
            resultado['termo1_custo_municao'],
            resultado['termo2_custo_falha'],
            resultado['custo_total_esperado'],
            '',
            n_min_custo,
            resultado_min['custo_total_esperado'],
            resultado_min['prob_sucesso'] * 100,
            resultado_min['num_esperado_disparos'],
        ]
    })
    resumo.to_excel(writer, sheet_name='Resumo', index=False)

print(f"✓ Resultados salvos: resultados_valor_esperado.xlsx")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print(f"\n{'='*80}")
print(f"✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
print(f"{'='*80}")

print(f"\n📋 ARQUIVOS GERADOS:")
print(f"{'─'*80}")
print(f"   1. probabilidade_sucesso.png")
print(f"   2. custo_esperado.png")
print(f"   3. custo_municao.png")
print(f"   4. resultados_valor_esperado.xlsx")

print(f"\n📊 RESUMO EXECUTIVO:")
print(f"{'─'*80}")
print(f"   Total de disparos analisados:  {n_total}")
print(f"   N ótimo (custo mínimo):        {n_min_custo}")
print(f"   Custo mínimo:                  USD {resultado_min['custo_total_esperado']:,.2f}")
print(f"   P(Sucesso) no ótimo:           {resultado_min['prob_sucesso']*100:.6f}%")
print(f"   Taxa de acerto média:          {df['Taxa_acerto_Drone_Sea_Baby_pct'].mean():.6f}%")

