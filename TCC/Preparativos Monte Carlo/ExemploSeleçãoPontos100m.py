# =============================================================================
# EXEMPLO DE USO - LEITURA E SELEÇÃO DE PONTOS COM DISTÂNCIA ~100m
# =============================================================================

if __name__ == "__main__":
    
    import time
    
    print("="*80)
    print("ANÁLISE DE PONTOS COM ESPAÇAMENTO DE ~100m")
    print("="*80)
    
    # =========================================================================
    # CARREGAR ARQUIVO EXCEL COM AZIMUTES ÓTIMOS
    # =========================================================================
    print("\nCarregando arquivo de azimutes ótimos...")
    excel_path = 'azimutes_otimos_deriva_zero.xlsx'
    
    try:
        df_otimos = pd.read_excel(excel_path, engine='openpyxl')
        print(f"✓ Arquivo carregado: {excel_path}")
        print(f"  Total de linhas: {len(df_otimos)}")
        print(f"  Colunas: {list(df_otimos.columns)}")
    except FileNotFoundError:
        print(f"✗ Erro: Arquivo '{excel_path}' não encontrado!")
        print("  Execute primeiro a varredura completa para gerar o arquivo.")
        exit(1)
    
    # =========================================================================
    # FILTRAR A PARTIR DE ELEVAÇÃO 39.6° ATÉ -15° E ORDENAR DECRESCENTE
    # =========================================================================
    print(f"\n{'='*80}")
    print("SELEÇÃO DE PONTOS COM ESPAÇAMENTO DE ~100m")
    print(f"{'='*80}")
    
    # Filtrar elevações entre -15° e 39.6°
    elevacao_inicial = 39.6
    elevacao_final = -15.0
    df_filtrado = df_otimos[(df_otimos['Elevacao_deg'] <= elevacao_inicial) & 
                             (df_otimos['Elevacao_deg'] >= elevacao_final)].copy()
    
    # Ordenar por elevação DECRESCENTE
    df_filtrado = df_filtrado.sort_values('Elevacao_deg', ascending=False)
    df_filtrado.reset_index(drop=True, inplace=True)
    
    print(f"  Elevação inicial: {elevacao_inicial}°")
    print(f"  Elevação final: {elevacao_final}°")
    print(f"  Total de pontos disponíveis: {len(df_filtrado)}")
    print(f"  Faixa de elevação: {df_filtrado['Elevacao_deg'].max():.1f}° a {df_filtrado['Elevacao_deg'].min():.1f}°")
    print(f"  Faixa de alcance: {df_filtrado['Alcance_x_m'].max():.1f} m a {df_filtrado['Alcance_x_m'].min():.1f} m")
    
    # =========================================================================
    # SELECIONAR PONTOS COM ~100m DE DIFERENÇA (MÉTODO ADAPTATIVO)
    # =========================================================================
    print(f"\nSelecionando pontos com espaçamento de ~100m...")
    
    espacamento_desejado = 100.0  # metros
    tolerancia_base = 20.0  # tolerância inicial de ±20m
    tolerancia_maxima = 50.0  # tolerância máxima de ±50m (para casos difíceis)
    
    # Listas para armazenar pontos selecionados
    pontos_selecionados = []
    
    # Começar com o primeiro ponto (maior elevação <= 39.6°)
    idx_atual = 0
    pontos_selecionados.append(df_filtrado.iloc[idx_atual].to_dict())
    alcance_anterior = df_filtrado.iloc[idx_atual]['Alcance_x_m']
    
    print(f"\n  Ponto inicial:")
    print(f"    Elevação: {df_filtrado.iloc[idx_atual]['Elevacao_deg']:.1f}°")
    print(f"    Alcance: {alcance_anterior:.1f} m")
    
    # Percorrer os demais pontos
    idx_busca = idx_atual + 1
    
    while idx_busca < len(df_filtrado):
        melhor_idx = None
        melhor_diferenca = None
        tolerancia_usada = tolerancia_base
        
        # Tentar encontrar ponto com tolerância base
        for idx in range(idx_busca, len(df_filtrado)):
            alcance_atual = df_filtrado.iloc[idx]['Alcance_x_m']
            diferenca = abs(alcance_anterior - alcance_atual)
            
            # Verificar se a diferença está na faixa desejada
            if abs(diferenca - espacamento_desejado) <= tolerancia_usada:
                melhor_idx = idx
                melhor_diferenca = diferenca
                break
        
        # Se não encontrou com tolerância base, tentar com tolerância maior
        if melhor_idx is None:
            tolerancia_usada = tolerancia_maxima
            
            # Procurar o ponto mais próximo de 100m
            for idx in range(idx_busca, len(df_filtrado)):
                alcance_atual = df_filtrado.iloc[idx]['Alcance_x_m']
                diferenca = abs(alcance_anterior - alcance_atual)
                erro = abs(diferenca - espacamento_desejado)
                
                if erro <= tolerancia_usada:
                    if melhor_idx is None or erro < abs(melhor_diferenca - espacamento_desejado):
                        melhor_idx = idx
                        melhor_diferenca = diferenca
        
        # Se ainda não encontrou, pegar o próximo ponto que tenha pelo menos 50m de diferença
        if melhor_idx is None:
            for idx in range(idx_busca, len(df_filtrado)):
                alcance_atual = df_filtrado.iloc[idx]['Alcance_x_m']
                diferenca = abs(alcance_anterior - alcance_atual)
                
                if diferenca >= 50:  # Pelo menos 50m
                    melhor_idx = idx
                    melhor_diferenca = diferenca
                    tolerancia_usada = None  # Marca como "fora da tolerância"
                    break
        
        # Se encontrou um ponto, adicionar
        if melhor_idx is not None:
            pontos_selecionados.append(df_filtrado.iloc[melhor_idx].to_dict())
            
            if tolerancia_usada is not None:
                print(f"  ✓ Ponto {len(pontos_selecionados)}:")
            else:
                print(f"  ⚠ Ponto {len(pontos_selecionados)} (tolerância expandida):")
            
            print(f"      Elevação: {df_filtrado.iloc[melhor_idx]['Elevacao_deg']:.1f}°")
            print(f"      Alcance: {df_filtrado.iloc[melhor_idx]['Alcance_x_m']:.1f} m")
            print(f"      Diferença: {melhor_diferenca:.1f} m")
            
            alcance_anterior = df_filtrado.iloc[melhor_idx]['Alcance_x_m']
            idx_busca = melhor_idx + 1
        else:
            # Se não encontrou nada, pular para o próximo
            idx_busca += 1
    
    # Converter para DataFrame
    df_selecionados = pd.DataFrame(pontos_selecionados)
    
    print(f"\n{'='*80}")
    print("RESULTADO DA SELEÇÃO")
    print(f"{'='*80}")
    print(f"  Total de pontos selecionados: {len(df_selecionados)}")
    print(f"  Faixa de elevação: {df_selecionados['Elevacao_deg'].max():.1f}° a {df_selecionados['Elevacao_deg'].min():.1f}°")
    print(f"  Faixa de alcance: {df_selecionados['Alcance_x_m'].max():.1f} m a {df_selecionados['Alcance_x_m'].min():.1f} m")
    
    # Verificar se chegou até -15°
    if df_selecionados['Elevacao_deg'].min() > elevacao_final:
        print(f"\n  ⚠ AVISO: Seleção parou em {df_selecionados['Elevacao_deg'].min():.1f}°")
        print(f"          Não foi possível chegar até {elevacao_final}°")
    else:
        print(f"\n  ✓ Seleção completa até {elevacao_final}°")
    
    # Calcular estatísticas das diferenças
    diferencas = []
    for i in range(1, len(df_selecionados)):
        dif = abs(df_selecionados.iloc[i]['Alcance_x_m'] - df_selecionados.iloc[i-1]['Alcance_x_m'])
        diferencas.append(dif)
    
    if diferencas:
        print(f"\n  ESTATÍSTICAS DAS DIFERENÇAS:")
        print(f"    Média: {np.mean(diferencas):.1f} m")
        print(f"    Mínima: {np.min(diferencas):.1f} m")
        print(f"    Máxima: {np.max(diferencas):.1f} m")
        print(f"    Desvio padrão: {np.std(diferencas):.1f} m")
    
    # =========================================================================
    # SALVAR PONTOS SELECIONADOS
    # =========================================================================
    print(f"\n{'='*80}")
    print("SALVANDO PONTOS SELECIONADOS")
    print(f"{'='*80}")
    
    # Adicionar coluna com diferença em relação ao ponto anterior
    df_selecionados['Diferenca_x_m'] = 0.0
    for i in range(1, len(df_selecionados)):
        df_selecionados.loc[df_selecionados.index[i], 'Diferenca_x_m'] = \
            abs(df_selecionados.iloc[i]['Alcance_x_m'] - df_selecionados.iloc[i-1]['Alcance_x_m'])
    
    # Salvar em Excel e CSV
    excel_output = 'pontos_selecionados_100m.xlsx'
    csv_output = 'pontos_selecionados_100m.csv'
    
    df_selecionados.to_excel(excel_output, index=False, engine='openpyxl')
    df_selecionados.to_csv(csv_output, index=False)
    
    print(f"✓ Pontos salvos em:")
    print(f"    • Excel: {excel_output}")
    print(f"    • CSV:   {csv_output}")
    
    # =========================================================================
    # MOSTRAR TABELA COMPLETA
    # =========================================================================
    print(f"\n{'='*80}")
    print("TABELA DE PONTOS SELECIONADOS")
    print(f"{'='*80}")
    print(f"{'#':>3s} | {'Elevação':>9s} | {'Azimute':>10s} | {'Alcance X':>11s} | {'Diferença':>10s} | {'Desvio Z':>9s}")
    print(f"{'-'*3}-+-{'-'*9}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*9}")
    
    for i in range(len(df_selecionados)):
        print(f"{i+1:3d} | {df_selecionados.iloc[i]['Elevacao_deg']:9.1f}° | "
              f"{df_selecionados.iloc[i]['Azimute_otimo_deg']:10.3f}° | "
              f"{df_selecionados.iloc[i]['Alcance_x_m']:11.1f} m | "
              f"{df_selecionados.iloc[i]['Diferenca_x_m']:10.1f} m | "
              f"{df_selecionados.iloc[i]['Desvio_z_resultante_m']:9.2f} m")
    
    # =========================================================================
    # VISUALIZAÇÃO DOS PONTOS SELECIONADOS
    # =========================================================================
    print(f"\n{'='*80}")
    print("GERANDO VISUALIZAÇÃO")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Alcance vs Elevação
    ax1 = axes[0, 0]
    ax1.plot(df_filtrado['Elevacao_deg'], df_filtrado['Alcance_x_m']/1000, 
             'b-', linewidth=1, alpha=0.3, label='Todos os pontos')
    ax1.scatter(df_selecionados['Elevacao_deg'], df_selecionados['Alcance_x_m']/1000,
                c='red', s=100, marker='o', edgecolors='black', linewidth=1.5,
                label='Pontos selecionados', zorder=5)
    
    # Adicionar linhas de conexão
    for i in range(1, len(df_selecionados)):
        ax1.plot([df_selecionados.iloc[i-1]['Elevacao_deg'], df_selecionados.iloc[i]['Elevacao_deg']],
                [df_selecionados.iloc[i-1]['Alcance_x_m']/1000, df_selecionados.iloc[i]['Alcance_x_m']/1000],
                'r--', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Elevação [°]', fontsize=12)
    ax1.set_ylabel('Alcance X [km]', fontsize=12)
    ax1.set_title('Alcance vs Elevação\n(Pontos com ~100m de espaçamento)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.axvline(x=elevacao_final, color='green', linestyle='--', linewidth=2, alpha=0.5, label=f'Alvo: {elevacao_final}°')
    
    # Gráfico 2: Diferenças entre pontos consecutivos
    ax2 = axes[0, 1]
    if len(df_selecionados) > 1:
        diferencas_plot = df_selecionados['Diferenca_x_m'].iloc[1:].values
        elevacoes_plot = df_selecionados['Elevacao_deg'].iloc[1:].values
        
        # Colorir barras baseado na conformidade
        cores = ['green' if abs(d - espacamento_desejado) <= tolerancia_base 
                else 'orange' if abs(d - espacamento_desejado) <= tolerancia_maxima
                else 'red' for d in diferencas_plot]
        
        ax2.bar(range(1, len(df_selecionados)), diferencas_plot, 
                color=cores, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.axhline(y=100, color='darkgreen', linestyle='--', linewidth=2, 
                   label='Espaçamento ideal (100m)', alpha=0.7)
        ax2.axhline(y=80, color='orange', linestyle=':', linewidth=1, 
                   label='Tolerância base (±20m)', alpha=0.5)
        ax2.axhline(y=120, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel('Índice do Ponto', fontsize=12)
        ax2.set_ylabel('Diferença de Alcance [m]', fontsize=12)
        ax2.set_title('Diferença de Alcance entre Pontos Consecutivos\n(Verde: ±20m | Laranja: ±50m | Vermelho: >50m)', 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=10)
    
    # Gráfico 3: Azimute Ótimo vs Elevação
    ax3 = axes[1, 0]
    ax3.plot(df_filtrado['Elevacao_deg'], df_filtrado['Azimute_otimo_deg'],
             'b-', linewidth=1, alpha=0.3, label='Todos os pontos')
    ax3.scatter(df_selecionados['Elevacao_deg'], df_selecionados['Azimute_otimo_deg'],
                c='red', s=100, marker='o', edgecolors='black', linewidth=1.5,
                label='Pontos selecionados', zorder=5)
    
    ax3.set_xlabel('Elevação [°]', fontsize=12)
    ax3.set_ylabel('Azimute Ótimo [°]', fontsize=12)
    ax3.set_title('Azimute Ótimo vs Elevação\n(para deriva ≈ 0)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax3.axvline(x=elevacao_final, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    # Gráfico 4: Vista de cima - Pontos de impacto
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_selecionados['Alcance_x_m']/1000, df_selecionados['Desvio_z_resultante_m'],
                c=df_selecionados['Elevacao_deg'], cmap='jet', s=150, 
                edgecolors='black', linewidth=1.5, marker='o')
    
    # Adicionar números aos pontos
    for i in range(len(df_selecionados)):
        ax4.text(df_selecionados.iloc[i]['Alcance_x_m']/1000, 
                df_selecionados.iloc[i]['Desvio_z_resultante_m'],
                f"{i+1}", fontsize=8, ha='center', va='center', 
                fontweight='bold', color='white')
    
    # Linha Z=0
    ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label='Deriva = 0')
    
    ax4.set_xlabel('Alcance X [km]', fontsize=12)
    ax4.set_ylabel('Desvio Lateral Z [m]', fontsize=12)
    ax4.set_title('Vista de Cima: Pontos de Impacto\n(coloridos por elevação)', 
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # Adicionar colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Elevação [°]', fontsize=10)
    
    plt.suptitle(f'Análise de Pontos Selecionados (Espaçamento ~100m)\n' +
                 f'Total: {len(df_selecionados)} pontos | Elevação: {df_selecionados["Elevacao_deg"].max():.1f}° a {df_selecionados["Elevacao_deg"].min():.1f}°',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pontos_selecionados_100m.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualização salva em: pontos_selecionados_100m.png")
    
    plt.show()
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    print(f"\n{'='*80}")
    print("RESUMO FINAL")
    print(f"{'='*80}")
    print(f"  Pontos totais disponíveis ({elevacao_inicial}° a {elevacao_final}°): {len(df_filtrado)}")
    print(f"  Pontos selecionados (espaçamento ~100m): {len(df_selecionados)}")
    print(f"  Taxa de seleção: {len(df_selecionados)/len(df_filtrado)*100:.1f}%")
    print(f"\n  PARÂMETROS DE SELEÇÃO:")
    print(f"    Espaçamento desejado: {espacamento_desejado:.0f} m")
    print(f"    Tolerância base: ±{tolerancia_base:.0f} m")
    print(f"    Tolerância máxima: ±{tolerancia_maxima:.0f} m")
    print(f"\n  ARQUIVOS GERADOS:")
    print(f"    1. {excel_output}")
    print(f"    2. {csv_output}")
    print(f"    3. pontos_selecionados_100m.png")
    print(f"\n✓ Análise concluída com sucesso!")
    print("="*80)