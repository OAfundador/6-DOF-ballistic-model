import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar resultados
file_path = 'C:\\Users\\DELL\\Desktop\\IC\\intervalos_confianca_monte_carlo.xlsx'
df = pd.read_excel(file_path, sheet_name='Todos_Intervalos')

# Converter para proporção
df['p_hat'] = df['Taxa_Acerto_%'] / 100.0
df['ic_inf_prop'] = df['IC_Inferior_%'] / 100.0
df['ic_sup_prop'] = df['IC_Superior_%'] / 100.0

# Filtrar os dois drones
df_sea_baby = df[df['Alvo'] == 'Drone_Sea_Baby'].copy()
df_iris = df[df['Alvo'] == 'IRIS_Paykan'].copy()

# Ordenar por alcance
df_sea_baby = df_sea_baby.sort_values('Alcance_m').reset_index(drop=True)
df_iris = df_iris.sort_values('Alcance_m').reset_index(drop=True)

# ==================================================================================
# GRÁFICO 1: DRONE SEA BABY (Pontos PRETO, IC VERMELHO)
# ==================================================================================
fig, ax = plt.subplots(figsize=(20, 8))

yerr_lower_sb = df_sea_baby['p_hat'] - df_sea_baby['ic_inf_prop']
yerr_upper_sb = df_sea_baby['ic_sup_prop'] - df_sea_baby['p_hat']

ax.errorbar(df_sea_baby.index, df_sea_baby['p_hat'],
            yerr=[yerr_lower_sb, yerr_upper_sb],
            fmt='o',
            color='black',           # Pontos pretos
            ecolor='#DC143C',        # IC vermelho (Crimson)
            markersize=5,
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.8,
            label='Drone Sea Baby')

ax.set_xlabel('Índice (ordenado por distância)', fontsize=14, fontweight='bold')
ax.set_ylabel('Probabilidade de Acerto (p̂)', fontsize=14, fontweight='bold')
ax.set_title('Drone Sea Baby - Probabilidades com IC (95%)\n' +
             'K=1000, σ²=0.25, z=1.96, Margem de Erro=±3.10%',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([-0.02, max(0.2, df_sea_baby['ic_sup_prop'].max() + 0.02)])
ax.set_xlim([-2, len(df_sea_baby) + 2])

# Linha de referência
ax.axhline(y=0.1, color='gray', linestyle='--', linewidth=1.5, alpha=0.4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Informações
textstr = f'N = {len(df_sea_baby)} pontos\n'
textstr += f'Média = {df_sea_baby["p_hat"].mean()*100:.2f}%\n'
textstr += f'Máximo = {df_sea_baby["p_hat"].max()*100:.2f}%'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('C:\\Users\\DELL\\Desktop\\IC\\grafico_1_drone_sea_baby2.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 1 salvo: grafico_1_drone_sea_baby.png")

# ==================================================================================
# GRÁFICO 2: IRIS PAYKAN (Pontos AZUL ESCURO, IC AZUL CLARO)
# ==================================================================================
fig, ax = plt.subplots(figsize=(20, 8))

yerr_lower_iris = df_iris['p_hat'] - df_iris['ic_inf_prop']
yerr_upper_iris = df_iris['ic_sup_prop'] - df_iris['p_hat']

ax.errorbar(df_iris.index, df_iris['p_hat'],
            yerr=[yerr_lower_iris, yerr_upper_iris],
            fmt='o',
            color='#00008B',         # Pontos azul escuro (DarkBlue)
            ecolor='#4169E1',        # IC azul royal (RoyalBlue)
            markersize=5,
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.8,
            label='IRIS Paykan')

ax.set_xlabel('Índice (ordenado por distância)', fontsize=14, fontweight='bold')
ax.set_ylabel('Probabilidade de Acerto (p̂)', fontsize=14, fontweight='bold')
ax.set_title('IRIS Paykan - Probabilidades com IC (95%)\n' +
             'K=1000, σ²=0.25, z=1.96, Margem de Erro=±3.10%',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([-0.02, min(1.02, df_iris['ic_sup_prop'].max() + 0.05)])
ax.set_xlim([-2, len(df_iris) + 2])

# Linhas de referência
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='50%')
ax.axhline(y=0.25, color='gray', linestyle=':', linewidth=1, alpha=0.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Informações
textstr = f'N = {len(df_iris)} pontos\n'
textstr += f'Média = {df_iris["p_hat"].mean()*100:.2f}%\n'
textstr += f'Máximo = {df_iris["p_hat"].max()*100:.2f}%'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('C:\\Users\\DELL\\Desktop\\IC\\grafico_2_iris_paykan2.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 2 salvo: grafico_2_iris_paykan.png")

# ==================================================================================
# GRÁFICO 3: AMBOS NO MESMO GRÁFICO
# ==================================================================================
fig, ax = plt.subplots(figsize=(22, 10))

# Plotar Drone Sea Baby
ax.errorbar(df_sea_baby.index, df_sea_baby['p_hat'],
            yerr=[yerr_lower_sb, yerr_upper_sb],
            fmt='o',
            color='black',
            ecolor='#DC143C',
            markersize=5,
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.7,
            label='Drone Sea Baby')

# Plotar IRIS Paykan
ax.errorbar(df_iris.index, df_iris['p_hat'],
            yerr=[yerr_lower_iris, yerr_upper_iris],
            fmt='s',                 # Quadrados para diferenciar
            color='#00008B',
            ecolor='#4169E1',
            markersize=5,
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.7,
            label='IRIS Paykan')

ax.set_xlabel('Índice (ordenado por distância)', fontsize=14, fontweight='bold')
ax.set_ylabel('Probabilidade de Acerto (p̂)', fontsize=14, fontweight='bold')
ax.set_title('Comparação: Drone Sea Baby vs IRIS Paykan - Probabilidades com IC (95%)\n' +
             'K=1000, σ²=0.25, z=1.96, Margem de Erro=±3.10%',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_ylim([-0.02, min(1.02, max(df_sea_baby['ic_sup_prop'].max(), df_iris['ic_sup_prop'].max()) + 0.05)])
ax.set_xlim([-2, max(len(df_sea_baby), len(df_iris)) + 2])

# Linhas de referência
ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, alpha=0.3, label='10%')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.3, label='50%')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Informações comparativas
textstr = f'Drone Sea Baby:\n'
textstr += f'  Média = {df_sea_baby["p_hat"].mean()*100:.2f}%\n'
textstr += f'  Máximo = {df_sea_baby["p_hat"].max()*100:.2f}%\n\n'
textstr += f'IRIS Paykan:\n'
textstr += f'  Média = {df_iris["p_hat"].mean()*100:.2f}%\n'
textstr += f'  Máximo = {df_iris["p_hat"].max()*100:.2f}%'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig('C:\\Users\\DELL\\Desktop\\IC\\grafico_3_comparacao_ambos2.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico 3 salvo: grafico_3_comparacao_ambos.png")

# ==================================================================================
# ESTATÍSTICAS COMPARATIVAS
# ==================================================================================
print("\n" + "="*80)
print("ESTATÍSTICAS COMPARATIVAS")
print("="*80)

print("\n" + "-"*80)
print("DRONE SEA BABY")
print("-"*80)
print(f"Pontos: {len(df_sea_baby)}")
print(f"Distância: {df_sea_baby['Alcance_m'].min():.1f}m - {df_sea_baby['Alcance_m'].max():.1f}m")
print(f"Probabilidade média: {df_sea_baby['p_hat'].mean()*100:.2f}%")
print(f"Probabilidade máxima: {df_sea_baby['p_hat'].max()*100:.2f}%")
print(f"Desvio padrão: {df_sea_baby['p_hat'].std()*100:.2f}%")
print(f"Mediana: {df_sea_baby['p_hat'].median()*100:.2f}%")

print("\n" + "-"*80)
print("IRIS PAYKAN")
print("-"*80)
print(f"Pontos: {len(df_iris)}")
print(f"Distância: {df_iris['Alcance_m'].min():.1f}m - {df_iris['Alcance_m'].max():.1f}m")
print(f"Probabilidade média: {df_iris['p_hat'].mean()*100:.2f}%")
print(f"Probabilidade máxima: {df_iris['p_hat'].max()*100:.2f}%")
print(f"Desvio padrão: {df_iris['p_hat'].std()*100:.2f}%")
print(f"Mediana: {df_iris['p_hat'].median()*100:.2f}%")

print("\n" + "-"*80)
print("COMPARAÇÃO")
print("-"*80)
ratio_media = df_iris['p_hat'].mean() / df_sea_baby['p_hat'].mean()
ratio_max = df_iris['p_hat'].max() / df_sea_baby['p_hat'].max()
print(f"IRIS Paykan tem taxa média {ratio_media:.1f}x maior que Drone Sea Baby")
print(f"IRIS Paykan tem taxa máxima {ratio_max:.1f}x maior que Drone Sea Baby")

# Contar pontos acima de 50%
n_sb_50 = len(df_sea_baby[df_sea_baby['p_hat'] > 0.5])
n_iris_50 = len(df_iris[df_iris['p_hat'] > 0.5])
print(f"\nPontos com p̂ > 50%:")
print(f"  Drone Sea Baby: {n_sb_50} ({n_sb_50/len(df_sea_baby)*100:.1f}%)")
print(f"  IRIS Paykan: {n_iris_50} ({n_iris_50/len(df_iris)*100:.1f}%)")

# Contar pontos acima de 10%
n_sb_10 = len(df_sea_baby[df_sea_baby['p_hat'] > 0.1])
n_iris_10 = len(df_iris[df_iris['p_hat'] > 0.1])
print(f"\nPontos com p̂ > 10%:")
print(f"  Drone Sea Baby: {n_sb_10} ({n_sb_10/len(df_sea_baby)*100:.1f}%)")
print(f"  IRIS Paykan: {n_iris_10} ({n_iris_10/len(df_iris)*100:.1f}%)")

print("\n" + "="*80)
print("3 GRÁFICOS GERADOS COM SUCESSO!")
print("="*80)
print("\nGráfico 1: Drone Sea Baby (preto/vermelho)")
print("Gráfico 2: IRIS Paykan (azul escuro/azul royal)")
print("Gráfico 3: Ambos juntos para comparação")
print("\nMarcadores: ● círculos (Sea Baby) | ■ quadrados (IRIS Paykan)")


