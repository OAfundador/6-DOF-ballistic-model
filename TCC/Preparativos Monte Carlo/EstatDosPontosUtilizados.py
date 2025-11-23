# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# =============================================================================
# PALETA DE CORES 3BLUE1BROWN - FUNDO BRANCO
# =============================================================================
COLORS = {
    'bg': '#FFFFFF',
    'primary': '#1A7FA0',
    'secondary': '#2E5F7D',
    'accent1': '#3BA3C7',
    'accent2': '#1A5F7A',
    'grid': '#D0D0D0',
    'text': '#1A1A1A',
    'green': '#2E7D32',
    'red': '#C62828',
    'yellow': '#F9A825',
    'purple': '#6A1B9A',
    'orange': '#EF6C00'
}

# =============================================================================
# CARREGAR DADOS
# =============================================================================
print("="*80)
print("VISUALIZACAO DOS PONTOS SELECIONADOS - ESTILO 3BLUE1BROWN")
print("="*80)

# CAMINHO DO ARQUIVO
caminho_arquivo = r'C:\Users\DELL\Downloads\pontos_selecionados_100m.xlsx'

try:
    df = pd.read_excel(caminho_arquivo, engine='openpyxl')
    print("\nArquivo carregado: {} pontos".format(len(df)))
    
    # INVERTER A ORDEM PARA FICAR CRESCENTE (menor alcance -> maior alcance)
    df = df.iloc[::-1].reset_index(drop=True)
    print("Ordem invertida: menor alcance ({:.2f} km) -> maior alcance ({:.2f} km)".format(
        df['Alcance_x_m'].iloc[0]/1000, df['Alcance_x_m'].iloc[-1]/1000))
    
except Exception as e:
    print("ERRO ao carregar arquivo: {}".format(e))
    exit(1)

# =============================================================================
# GRAFICO 1: VISTA DE CIMA (ALCANCE X vs DESVIO LATERAL Z)
# =============================================================================
print("\nGerando Grafico 1: Vista de Cima...")

fig1, ax1 = plt.subplots(figsize=(16, 10), facecolor=COLORS['bg'])
ax1.set_facecolor(COLORS['bg'])

# Configurar estilo
ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.8)
ax1.tick_params(colors=COLORS['text'], labelsize=13)
for spine in ax1.spines.values():
    spine.set_color(COLORS['text'])
    spine.set_linewidth(1.5)

# Linha de trajetoria
ax1.plot(df['Alcance_x_m']/1000, df['Desvio_z_resultante_m'],
        color=COLORS['primary'], linewidth=3, alpha=0.5, 
        linestyle='-', zorder=2)

# Pontos com gradiente de cores
scatter = ax1.scatter(df['Alcance_x_m']/1000, df['Desvio_z_resultante_m'],
                     c=np.arange(len(df)), cmap='viridis', s=200,
                     edgecolors=COLORS['text'], linewidths=2, 
                     marker='o', zorder=4, alpha=0.85)

# Linha de referencia Z=0
ax1.axhline(y=0, color=COLORS['green'], linestyle='--', linewidth=2.5,
           alpha=0.7, zorder=3)

# Preencher areas
ax1.fill_between(df['Alcance_x_m']/1000, 0, df['Desvio_z_resultante_m'],
                where=(df['Desvio_z_resultante_m'] >= 0),
                color=COLORS['green'], alpha=0.12, interpolate=True)
ax1.fill_between(df['Alcance_x_m']/1000, 0, df['Desvio_z_resultante_m'],
                where=(df['Desvio_z_resultante_m'] < 0),
                color=COLORS['red'], alpha=0.12, interpolate=True)

# Destacar primeiro e ultimo ponto
ax1.scatter([df.iloc[0]['Alcance_x_m']/1000], 
           [df.iloc[0]['Desvio_z_resultante_m']],
           c=COLORS['green'], s=400, marker='o', 
           edgecolors=COLORS['text'], linewidths=3, zorder=6)
ax1.scatter([df.iloc[-1]['Alcance_x_m']/1000], 
           [df.iloc[-1]['Desvio_z_resultante_m']],
           c=COLORS['red'], s=400, marker='s', 
           edgecolors=COLORS['text'], linewidths=3, zorder=6)

# Labels
ax1.set_xlabel('Alcance [km]', fontsize=16, fontweight='bold', color=COLORS['text'])
ax1.set_ylabel('Desvio Lateral [m]', fontsize=16, fontweight='bold', color=COLORS['text'])
ax1.set_title('Vista de Cima: Pontos de Impacto', 
             fontsize=18, fontweight='bold', color=COLORS['text'], pad=20)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
cbar.set_label('Sequencia de Pontos', fontsize=13, fontweight='bold', color=COLORS['text'])
cbar.ax.tick_params(colors=COLORS['text'], labelsize=11)
cbar.outline.set_edgecolor(COLORS['text'])
cbar.outline.set_linewidth(1.5)

# Ajustar limites
x_margin = (df['Alcance_x_m'].max() - df['Alcance_x_m'].min()) * 0.08 / 1000
y_max = max(abs(df['Desvio_z_resultante_m'].max()), 
            abs(df['Desvio_z_resultante_m'].min()))
y_margin = y_max * 0.15
ax1.set_xlim([df['Alcance_x_m'].min()/1000 - x_margin, 
              df['Alcance_x_m'].max()/1000 + x_margin])
ax1.set_ylim([-y_max - y_margin, y_max + y_margin])

plt.tight_layout()
plt.savefig('grafico_1_vista_cima.png', dpi=300, facecolor=COLORS['bg'], bbox_inches='tight')
print("Salvo: grafico_1_vista_cima.png")
plt.show()

# =============================================================================
# GRAFICO 2: DISTANCIAS ENTRE PONTOS CONSECUTIVOS
# =============================================================================
print("\nGerando Grafico 2: Distancias entre pontos...")

if len(df) > 1:
    # Calcular distancias (agora serao positivas)
    diferencas = df['Alcance_x_m'].diff().dropna().values
    indices = np.arange(1, len(df))
    
    print("Distancias calculadas: min={:.1f}m, max={:.1f}m, media={:.1f}m".format(
        diferencas.min(), diferencas.max(), diferencas.mean()))
    
    fig2, ax2 = plt.subplots(figsize=(16, 9), facecolor=COLORS['bg'])
    ax2.set_facecolor(COLORS['bg'])
    
    # Configurar estilo
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.8, axis='y')
    ax2.tick_params(colors=COLORS['text'], labelsize=13)
    for spine in ax2.spines.values():
        spine.set_color(COLORS['text'])
        spine.set_linewidth(1.5)
    
    # Determinar cores baseadas na proximidade de 100m
    cores = []
    for d in diferencas:
        if 80 <= d <= 120:  # Dentro de +/- 20m
            cores.append(COLORS['green'])
        elif 70 <= d <= 130:  # Dentro de +/- 30m
            cores.append(COLORS['yellow'])
        else:
            cores.append(COLORS['red'])
    
    # Barras
    bars = ax2.bar(indices, diferencas, width=0.7, color=cores, 
                   alpha=0.8, edgecolor=COLORS['text'], linewidth=1.5)
    
    # Linha de referencia (100m ideal)
    ax2.axhline(y=100, color=COLORS['primary'], linestyle='--', 
               linewidth=3, alpha=0.8, zorder=3)
    
    # Faixa de tolerancia
    ax2.fill_between([0, len(df)], 80, 120, 
                     color=COLORS['green'], alpha=0.15, zorder=1)
    
    # Linha conectando os valores
    ax2.plot(indices, diferencas, color=COLORS['secondary'], 
            linewidth=2.5, alpha=0.6, marker='o', markersize=8,
            markerfacecolor=COLORS['accent1'], markeredgecolor=COLORS['text'],
            markeredgewidth=1.5, zorder=4)
    
    # Labels
    ax2.set_xlabel('Intervalo entre Pontos', fontsize=16, fontweight='bold', color=COLORS['text'])
    ax2.set_ylabel('Distancia [m]', fontsize=16, fontweight='bold', color=COLORS['text'])
    ax2.set_title('Espacamento entre Pontos Consecutivos', 
                 fontsize=18, fontweight='bold', color=COLORS['text'], pad=20)
    
    # Limites
    ax2.set_xlim([0.5, len(df) - 0.5])
    y_min = min(diferencas.min(), 70)
    y_max = max(diferencas.max(), 130)
    y_range = y_max - y_min
    ax2.set_ylim([y_min - y_range*0.1, y_max + y_range*0.1])
    
    plt.tight_layout()
    plt.savefig('grafico_2_distancias.png', dpi=300, facecolor=COLORS['bg'], bbox_inches='tight')
    print("Salvo: grafico_2_distancias.png")
    plt.show()

# =============================================================================
# GRAFICO 3: HISTOGRAMA DAS DISTANCIAS
# =============================================================================
print("\nGerando Grafico 3: Histograma das distancias...")

if len(df) > 1:
    fig3, ax3 = plt.subplots(figsize=(14, 9), facecolor=COLORS['bg'])
    ax3.set_facecolor(COLORS['bg'])
    
    # Configurar estilo
    ax3.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.8, axis='y')
    ax3.tick_params(colors=COLORS['text'], labelsize=13)
    for spine in ax3.spines.values():
        spine.set_color(COLORS['text'])
        spine.set_linewidth(1.5)
    
    # Criar histograma
    n_bins = min(20, len(diferencas) // 2) if len(diferencas) > 4 else 10
    counts, bins, patches = ax3.hist(diferencas, bins=n_bins, 
                                     color=COLORS['primary'], alpha=0.7,
                                     edgecolor=COLORS['text'], linewidth=1.5,
                                     zorder=3)
    
    # Colorir barras baseado na posicao
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if 80 <= bin_center <= 120:
            patch.set_facecolor(COLORS['green'])
        elif 70 <= bin_center <= 130:
            patch.set_facecolor(COLORS['yellow'])
        else:
            patch.set_facecolor(COLORS['red'])
    
    # Linha vertical na media
    media = diferencas.mean()
    ax3.axvline(x=media, color=COLORS['purple'], linestyle='-', 
               linewidth=3, alpha=0.8, zorder=4,
               label='Media: {:.1f} m'.format(media))
    
    # Linha vertical no alvo (100m)
    ax3.axvline(x=100, color=COLORS['primary'], linestyle='--', 
               linewidth=3, alpha=0.8, zorder=4,
               label='Alvo: 100 m')
    
    # Faixas de tolerancia
    ax3.axvspan(80, 120, color=COLORS['green'], alpha=0.1, zorder=1,
               label='Tolerancia +/-20m')
    ax3.axvspan(70, 80, color=COLORS['yellow'], alpha=0.1, zorder=1)
    ax3.axvspan(120, 130, color=COLORS['yellow'], alpha=0.1, zorder=1)
    
    # Curva de densidade (KDE)
    from scipy import stats
    kde = stats.gaussian_kde(diferencas)
    x_kde = np.linspace(diferencas.min(), diferencas.max(), 200)
    y_kde = kde(x_kde)
    # Escalar para o histograma
    y_kde_scaled = y_kde * len(diferencas) * (bins[1] - bins[0])
    ax3.plot(x_kde, y_kde_scaled, color=COLORS['secondary'], 
            linewidth=3, alpha=0.8, zorder=5, label='Densidade')
    
    # Labels
    ax3.set_xlabel('Distancia entre Pontos [m]', fontsize=16, fontweight='bold', color=COLORS['text'])
    ax3.set_ylabel('Frequencia', fontsize=16, fontweight='bold', color=COLORS['text'])
    ax3.set_title('Distribuicao das Distancias entre Pontos Consecutivos', 
                 fontsize=18, fontweight='bold', color=COLORS['text'], pad=20)
    
    # Legenda
    legend = ax3.legend(facecolor=COLORS['bg'], edgecolor=COLORS['text'],
                       fontsize=12, loc='best', framealpha=1)
    for text in legend.get_texts():
        text.set_color(COLORS['text'])
    
    # Adicionar estatisticas no grafico
    stats_text = 'N = {}\nMedia = {:.1f} m\nDesvio Padrao = {:.1f} m\nMin = {:.1f} m\nMax = {:.1f} m'.format(
        len(diferencas), media, diferencas.std(), diferencas.min(), diferencas.max())
    
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=COLORS['bg'], 
                     edgecolor=COLORS['text'], alpha=0.9, linewidth=1.5),
            color=COLORS['text'], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('grafico_3_histograma.png', dpi=300, facecolor=COLORS['bg'], bbox_inches='tight')
    print("Salvo: grafico_3_histograma.png")
    plt.show()

# =============================================================================
# ESTATISTICAS
# =============================================================================
print("\n" + "="*80)
print("ESTATISTICAS")
print("="*80)
print("Total de pontos: {}".format(len(df)))
print("Alcance: {:.2f} km a {:.2f} km".format(
    df['Alcance_x_m'].min()/1000, df['Alcance_x_m'].max()/1000))
print("Desvio lateral: {:.2f} m a {:.2f} m".format(
    df['Desvio_z_resultante_m'].min(), df['Desvio_z_resultante_m'].max()))

if len(df) > 1:
    diferencas_df = df['Alcance_x_m'].diff().dropna()
    print("\nEspacamento entre pontos:")
    print("  Media: {:.1f} m".format(diferencas_df.mean()))
    print("  Mediana: {:.1f} m".format(diferencas_df.median()))
    print("  Minimo: {:.1f} m".format(diferencas_df.min()))
    print("  Maximo: {:.1f} m".format(diferencas_df.max()))
    print("  Desvio padrao: {:.1f} m".format(diferencas_df.std()))
    
    dentro_tolerancia = ((diferencas_df >= 80) & (diferencas_df <= 120)).sum()
    print("  Pontos dentro de +/-20m de 100m: {}/{} ({:.1f}%)".format(
        dentro_tolerancia, len(diferencas_df), 
        100 * dentro_tolerancia / len(diferencas_df)))

print("\n" + "="*80)
print("CONCLUIDO!")
