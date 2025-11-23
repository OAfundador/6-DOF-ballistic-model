# =============================================================================
# SIMULAÇÃO MONTE CARLO MASSIVA - MÚLTIPLAS EMBARCAÇÕES
# MODIFICADO: α=0, β=0 FIXOS | PERTURBAÇÕES: Elevação σ=0.1°, Azimute σ=0.05°
# ORDEM: ELEVAÇÃO DECRESCENTE (maior alcance até -1.5°)
# =============================================================================

if __name__ == "__main__":
    
    import time
    import datetime
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print("="*80)
    print("SIMULAÇÃO MONTE CARLO MASSIVA - MÚLTIPLAS EMBARCAÇÕES")
    print("ORDEM: ELEVAÇÃO DECRESCENTE (Máximo → -1.5°)")
    print("TOLERÂNCIAS: ORIGINAIS (rtol=1e-7, atol=1e-8)")
    print("MODIFICADO: α=0°, β=0° (FIXOS)")
    print("PERTURBAÇÕES: Elevação σ=0.1°, Azimute σ=0.05°")
    print("="*80)
    
    # =========================================================================
    # CARREGAR PONTOS SELECIONADOS
    # =========================================================================
    print("\nCarregando pontos selecionados...")
    excel_selecionados = r'C:\Users\DELL\Downloads\pontos_selecionados_100m.xlsx'
    
    try:
        df_pontos = pd.read_excel(excel_selecionados, engine='openpyxl')
        print(f"✓ Arquivo carregado: {excel_selecionados}")
        print(f"  Total de pontos: {len(df_pontos)}")
    except FileNotFoundError:
        print(f"✗ Erro: Arquivo '{excel_selecionados}' não encontrado!")
        print("  Execute primeiro a seleção de pontos.")
        exit(1)
    
    # =========================================================================
    # FILTRAR PONTOS: MAIOR ALCANCE ATÉ ELEVAÇÃO -1.5°
    # =========================================================================
    print("\n" + "="*80)
    print("FILTRANDO PONTOS PARA SIMULAÇÃO")
    print("="*80)
    
    # Encontrar ponto de maior alcance
    idx_max_alcance = df_pontos['Alcance_x_m'].idxmax()
    elevacao_max_alcance = df_pontos.loc[idx_max_alcance, 'Elevacao_deg']
    
    # Filtrar pontos entre elevação máxima e -1.5°
    elevacao_limite_inferior = -1.5
    df_pontos_filtrados = df_pontos[
        (df_pontos['Elevacao_deg'] <= elevacao_max_alcance) &
        (df_pontos['Elevacao_deg'] >= elevacao_limite_inferior)
    ].copy()
    
    # Ordenar por elevação DECRESCENTE (maior para menor)
    df_pontos_filtrados = df_pontos_filtrados.sort_values('Elevacao_deg', ascending=False)
    df_pontos_filtrados.reset_index(drop=True, inplace=True)
    
    n_pontos_total = len(df_pontos_filtrados)
    
    print(f"\n  Pontos selecionados para simulação:")
    print(f"    Elevação INICIAL: {df_pontos_filtrados.iloc[0]['Elevacao_deg']:.1f}° (maior)")
    print(f"    Elevação FINAL: {df_pontos_filtrados.iloc[-1]['Elevacao_deg']:.1f}° (menor)")
    print(f"    Total de pontos: {n_pontos_total}")
    print(f"    Alcance inicial: {df_pontos_filtrados.iloc[0]['Alcance_x_m']:.1f} m")
    print(f"    Alcance final: {df_pontos_filtrados.iloc[-1]['Alcance_x_m']:.1f} m")
    print(f"    ORDEM: Decrescente (ângulos positivos → negativos)")
    
    # =========================================================================
    # CONFIGURAÇÃO DO SIMULADOR
    # =========================================================================
    print("\n" + "="*80)
    print("CONFIGURANDO SIMULADOR")
    print("="*80)
    
    # NOTA: Aqui você deve importar suas classes
    # from seu_modulo import RealAerodynamicCoefficients, Projectile, Weapon, Environment, BallisticSimulator, Vessel
    
    # Carregar coeficientes aerodinâmicos
    aero_coeffs = RealAerodynamicCoefficients()
    
    # Criar projétil (Naval 5"/38)
    projectile = Projectile.from_imperial(
        name="Projétil Naval 5\"/38",
        mass_lb=68.10,
        diameter_in=5.0,
        I_P_lbin2=240.9,
        I_T_lbin2=2619.0,
        rifling_twist_calibers=25.0
    )
    
    # Criar arma em terra
    weapon = Weapon(
        name="Canhão Naval 5\"/38",
        position=(0.0, 10.0, 0.0),
        elevation_deg=45.0,
        azimuth_deg=0.0,
        rate_of_fire_rpm=15.0,
        muzzle_velocity_mps=807.0,
        mounted_on_vessel=None
    )
    
    # Criar ambiente
    environment = Environment(
        rho=1.225,
        g=9.81,
        W1=0.0,
        W2=0.0,
        W3=0.0
    )
    
    # Criar simulador
    simulator = BallisticSimulator(
        projectile=projectile,
        weapon=weapon,
        environment=environment,
        aero_coeffs=aero_coeffs
    )
    
    # =========================================================================
    # PARÂMETROS MONTE CARLO (MODIFICADO)
    # =========================================================================
    n_simulacoes_por_ponto = 1000
    
    # PERTURBAÇÕES NOS ÂNGULOS DE TIRO
    sigma_elevacao = 0.1  # ← Nova perturbação: elevação
    sigma_azimute = 0.05   # ← Nova perturbação: azimute
    
    # ALPHA E BETA FIXOS EM ZERO
    alpha0_deg_fixo = 0.0
    beta0_deg_fixo = 0.0
    
    seed_mc = 16184331
    
    n_simulacoes_total = n_pontos_total * n_simulacoes_por_ponto
    
    print(f"\n{'='*80}")
    print(f"CONFIGURAÇÃO MONTE CARLO MASSIVA (MODIFICADO)")
    print(f"{'='*80}")
    print(f"  Pontos a simular: {n_pontos_total}")
    print(f"  Simulações por ponto: {n_simulacoes_por_ponto}")
    print(f"  Total de simulações: {n_simulacoes_total:,}")
    print(f"  ")
    print(f"  MODIFICAÇÃO APLICADA:")
    print(f"    • α (alpha) = {alpha0_deg_fixo}° (FIXO)")
    print(f"    • β (beta) = {beta0_deg_fixo}° (FIXO)")
    print(f"    • Perturbação ELEVAÇÃO: Normal(μ=0, σ={sigma_elevacao}°)")
    print(f"    • Perturbação AZIMUTE: Normal(μ=0, σ={sigma_azimute}°)")
    print(f"  ")
    print(f"  Seed: {seed_mc}")
    print(f"  Backup automático: A cada 30 minutos")
    print(f"  TOLERÂNCIAS ORIGINAIS:")
    print(f"    • rtol=1e-7, atol=1e-8 (máxima precisão)")
    print(f"    • Sem otimizações de velocidade")
    
    # =========================================================================
    # DEFINIR MÚLTIPLAS EMBARCAÇÕES ALVO
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"DEFININDO EMBARCAÇÕES ALVO")
    print(f"{'='*80}")
    
    # Baseado nos links da Wikipedia (dimensões em metros)
    embarcacoes_specs = {
        'Drone_Sea_Baby': {
            'length': 6.0,
            'width': 2.0,
            'description': 'Drone naval ucraniano'
        },
        'IRIS_Paykan': {
            'length': 56.0,  # Fast attack craft iraniano
            'width': 7.6,
            'description': 'Fast attack craft (Irã)'
        },
        'Osa_class': {
            'length': 38.6,  # Osa-class missile boat
            'width': 7.6,
            'description': 'Osa-class missile boat (URSS)'
        },
        'Hayabusa_class': {
            'length': 50.1,  # Hayabusa-class torpedo boat
            'width': 8.4,
            'description': 'Hayabusa-class torpedo boat (Japão)'
        },
        'SMS_V4': {
            'length': 72.0,  # SMS V4 destroyer alemão WWI
            'width': 7.34,
            'description': 'SMS V4 destroyer (Alemanha WWI)'
        },
        'PT_105': {
            'length': 24.4,  # PT-105 patrol torpedo boat
            'width': 6.3,
            'description': 'PT-105 patrol torpedo boat (EUA)'
        }
    }
    
    print(f"\n  Embarcações configuradas: {len(embarcacoes_specs)}")
    for nome, specs in embarcacoes_specs.items():
        print(f"    • {nome}: {specs['length']}m × {specs['width']}m - {specs['description']}")
    
    # =========================================================================
    # GERAR TODAS AS PERTURBAÇÕES DE UMA VEZ
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"GERANDO TODAS AS PERTURBAÇÕES (ELEVAÇÃO E AZIMUTE)")
    print(f"{'='*80}")
    
    np.random.seed(seed_mc)
    
    # Gerar perturbações para ELEVAÇÃO e AZIMUTE (não mais alpha e beta)
    print(f"  Gerando {n_simulacoes_total * 2:,} valores aleatórios...")
    tempo_inicio_geracao = time.time()
    
    # Perturbações nos ângulos de tiro
    delta_elevacao_todas = np.random.normal(0, sigma_elevacao, n_simulacoes_total)
    delta_azimute_todas = np.random.normal(0, sigma_azimute, n_simulacoes_total)
    
    tempo_geracao = time.time() - tempo_inicio_geracao
    
    print(f"✓ Perturbações geradas em {tempo_geracao:.2f}s!")
    print(f"  ΔElevação: μ={delta_elevacao_todas.mean():.6f}°, σ={delta_elevacao_todas.std():.6f}°")
    print(f"  ΔAzimute: μ={delta_azimute_todas.mean():.6f}°, σ={delta_azimute_todas.std():.6f}°")
    print(f"  Memória ocupada: ~{(delta_elevacao_todas.nbytes + delta_azimute_todas.nbytes) / (1024**2):.2f} MB")
    
    # =========================================================================
    # PREPARAR ARMAZENAMENTO DE RESULTADOS
    # =========================================================================
    resultados_resumo = []
    
    # Arquivos de saída (com novos nomes refletindo as mudanças)
    arquivo_final = 'segundomonte_carlo_massivo_alpha0_beta0_elev03_azi01.xlsx'
    arquivo_backup = 'segundomonte_carlo_massivo_alpha0_beta0_elev03_azi01_backup.xlsx'
    
    # =========================================================================
    # LOOP PRINCIPAL: SIMULAR TODOS OS PONTOS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"INICIANDO SIMULAÇÃO MASSIVA (α=0, β=0, ΔElev σ=0.3°, ΔAzi σ=0.1°)")
    print(f"{'='*80}")
    
    tempo_inicio_total = time.time()
    tempo_ultimo_backup = time.time()
    intervalo_backup = 30 * 60  # 30 minutos em segundos
    
    contador_sim_global = 0
    
    for idx_ponto in range(n_pontos_total):
        ponto = df_pontos_filtrados.iloc[idx_ponto]
        
        elevacao_nominal = ponto['Elevacao_deg']
        azimute_nominal = ponto['Azimute_otimo_deg']
        alcance_nominal = ponto['Alcance_x_m']
        desvio_z_nominal = ponto['Desvio_z_resultante_m']
        numero_ponto = idx_ponto + 1
        
        print(f"\n{'─'*80}")
        print(f"PONTO {numero_ponto}/{n_pontos_total} | " +
              f"Elevação: {elevacao_nominal:.1f}° | " +
              f"Alcance: {alcance_nominal:.0f}m")
        print(f"{'─'*80}")
        
        # Criar TODAS as embarcações no mesmo ponto de impacto nominal
        embarcacoes_alvo = {}
        for nome_emb, specs in embarcacoes_specs.items():
            embarcacoes_alvo[nome_emb] = Vessel(
                name=nome_emb,
                center_position=(alcance_nominal, desvio_z_nominal),
                length=specs['length'],
                width=specs['width'],
                height=1.0,  # Altura ignorada na verificação
                velocity=(0.0, 0.0)
            )
        
        # Arrays para resultados deste ponto (para cada embarcação)
        acertos_por_embarcacao = {nome: 0 for nome in embarcacoes_specs.keys()}
        erros_x_ponto = []
        erros_z_ponto = []
        distancias_erro_ponto = []
        tempos_voo_ponto = []
        n_validas_ponto = 0
        
        tempo_inicio_ponto = time.time()
        
        # Simular 400 vezes para este ponto
        for sim in range(n_simulacoes_por_ponto):
            # Pegar perturbações pré-geradas para os ÂNGULOS DE TIRO
            delta_elevacao = delta_elevacao_todas[contador_sim_global]
            delta_azimute = delta_azimute_todas[contador_sim_global]
            contador_sim_global += 1
            
            # Calcular ângulos perturbados
            elevacao_perturbada = elevacao_nominal + delta_elevacao
            azimute_perturbado = azimute_nominal + delta_azimute
            
            # Configurar ângulos de tiro PERTURBADOS
            weapon.set_firing_angles(
                elevation_deg=elevacao_perturbada,
                azimuth_deg=azimute_perturbado
            )
            
            try:
                # SIMULAÇÃO COM ALPHA E BETA FIXOS EM ZERO
                result = simulator.simulate(
                    max_time=100.0,
                    alpha0_deg=alpha0_deg_fixo,  # ← FIXO EM 0
                    beta0_deg=beta0_deg_fixo,    # ← FIXO EM 0
                    w_j0=5.0,
                    w_k0=5.0,
                    rtol=1e-7,  # ← ORIGINAL (máxima precisão)
                    atol=1e-8   # ← ORIGINAL (máxima precisão)
                )
                
                # Posição final
                x_final = result.x[-1]
                y_final = result.y[-1]
                z_final = result.z[-1]
                posicao_impacto = np.array([x_final, y_final, z_final])
                
                # Verificar acerto em CADA embarcação
                for nome_emb, vessel in embarcacoes_alvo.items():
                    acertou = vessel.check_impact(posicao_impacto, time=result.t[-1], check_height=False)
                    if acertou:
                        acertos_por_embarcacao[nome_emb] += 1
                
                # Calcular erros (independente de qual embarcação)
                erro_x = x_final - alcance_nominal
                erro_z = z_final - desvio_z_nominal
                distancia_erro = np.sqrt(erro_x**2 + erro_z**2)
                
                # Armazenar
                erros_x_ponto.append(erro_x)
                erros_z_ponto.append(erro_z)
                distancias_erro_ponto.append(distancia_erro)
                tempos_voo_ponto.append(result.tempo_voo)
                n_validas_ponto += 1
                
            except Exception as e:
                # Simulação falhou, apenas continuar
                pass
            
            # Print de progresso a cada 50 simulações
            if (sim + 1) % 50 == 0:
                tempo_ponto_decorrido = time.time() - tempo_inicio_ponto
                tempo_por_sim = tempo_ponto_decorrido / (sim + 1)
                tempo_restante_ponto = tempo_por_sim * (n_simulacoes_por_ponto - (sim + 1))
                
                print(f"    [{sim+1}/{n_simulacoes_por_ponto}] " +
                      f"Restante (ponto): {tempo_restante_ponto:.1f}s | " +
                      f"Acertos Drone: {acertos_por_embarcacao['Drone_Sea_Baby']}")
        
        tempo_total_ponto = time.time() - tempo_inicio_ponto
        
        # Calcular estatísticas do ponto
        if n_validas_ponto > 0:
            erros_x_array = np.array(erros_x_ponto)
            erros_z_array = np.array(erros_z_ponto)
            distancias_array = np.array(distancias_erro_ponto)
            tempos_array = np.array(tempos_voo_ponto)
            
            # Criar dicionário de resumo com taxas de acerto para cada embarcação
            resumo_ponto = {
                'Ponto_numero': numero_ponto,
                'Elevacao_deg': elevacao_nominal,
                'Azimute_deg': azimute_nominal,
                'Alcance_m': alcance_nominal,
                'Desvio_Z_nominal_m': desvio_z_nominal,
                'N_simulacoes': n_simulacoes_por_ponto,
                'N_validas': n_validas_ponto,
                'Erro_X_medio_m': erros_x_array.mean(),
                'Erro_X_std_m': erros_x_array.std(),
                'Erro_X_min_m': erros_x_array.min(),
                'Erro_X_max_m': erros_x_array.max(),
                'Erro_Z_medio_m': erros_z_array.mean(),
                'Erro_Z_std_m': erros_z_array.std(),
                'Erro_Z_min_m': erros_z_array.min(),
                'Erro_Z_max_m': erros_z_array.max(),
                'CEP50_m': np.median(distancias_array),
                'CEP90_m': np.percentile(distancias_array, 90),
                'CEP95_m': np.percentile(distancias_array, 95),
                'Tempo_voo_medio_s': tempos_array.mean(),
                'Tempo_simulacao_s': tempo_total_ponto
            }
            
            # Adicionar taxas de acerto para cada embarcação
            for nome_emb, n_acertos in acertos_por_embarcacao.items():
                taxa = (n_acertos / n_validas_ponto) * 100
                resumo_ponto[f'Acertos_{nome_emb}'] = n_acertos
                resumo_ponto[f'Taxa_acerto_{nome_emb}_pct'] = taxa
            
            resultados_resumo.append(resumo_ponto)
            
            print(f"\n  ✓ Ponto concluído em {tempo_total_ponto:.1f}s")
            print(f"    CEP50: {np.median(distancias_array):.2f}m | CEP90: {np.percentile(distancias_array, 90):.2f}m")
            print(f"    Taxas de acerto:")
            for nome_emb in embarcacoes_specs.keys():
                taxa = (acertos_por_embarcacao[nome_emb] / n_validas_ponto) * 100
                print(f"      • {nome_emb}: {taxa:.1f}% ({acertos_por_embarcacao[nome_emb]}/{n_validas_ponto})")
        
        # Estimativa de tempo restante
        tempo_total_decorrido = time.time() - tempo_inicio_total
        pontos_restantes = n_pontos_total - (idx_ponto + 1)
        if idx_ponto > 0:
            tempo_medio_por_ponto = tempo_total_decorrido / (idx_ponto + 1)
            tempo_estimado_restante = tempo_medio_por_ponto * pontos_restantes
            
            print(f"\n  Progresso global: {numero_ponto}/{n_pontos_total} pontos ({(numero_ponto/n_pontos_total)*100:.1f}%)")
            print(f"  Tempo decorrido: {tempo_total_decorrido/60:.1f} min")
            print(f"  Tempo estimado restante: {tempo_estimado_restante/60:.1f} min")
            print(f"  Tempo total estimado: {(tempo_total_decorrido + tempo_estimado_restante)/60:.1f} min")
        
        # =====================================================================
        # BACKUP AUTOMÁTICO A CADA 30 MINUTOS
        # =====================================================================
        tempo_desde_ultimo_backup = time.time() - tempo_ultimo_backup
        
        if tempo_desde_ultimo_backup >= intervalo_backup:
            print(f"\n  {'='*76}")
            print(f"  SALVANDO BACKUP AUTOMÁTICO ({datetime.datetime.now().strftime('%H:%M:%S')})")
            print(f"  {'='*76}")
            
            df_backup = pd.DataFrame(resultados_resumo)
            df_backup.to_excel(arquivo_backup, index=False, engine='openpyxl')
            df_backup.to_csv(arquivo_backup.replace('.xlsx', '.csv'), index=False)
            
            print(f"  ✓ Backup salvo: {arquivo_backup}")
            print(f"    Pontos salvos: {len(resultados_resumo)}/{n_pontos_total}")
            
            tempo_ultimo_backup = time.time()
    
    # =========================================================================
    # SALVAR RESULTADOS FINAIS
    # =========================================================================
    tempo_total_final = time.time() - tempo_inicio_total
    
    print(f"\n{'='*80}")
    print(f"SIMULAÇÃO MASSIVA CONCLUÍDA!")
    print(f"{'='*80}")
    print(f"  Tempo total: {tempo_total_final/60:.1f} minutos ({tempo_total_final/3600:.2f} horas)")
    print(f"  Pontos simulados: {n_pontos_total}")
    print(f"  Simulações totais: {n_simulacoes_total:,}")
    print(f"  Tempo médio por ponto: {tempo_total_final/n_pontos_total:.1f}s")
    print(f"  Tempo médio por simulação: {tempo_total_final/n_simulacoes_total:.3f}s")
    
    print(f"\n{'='*80}")
    print(f"SALVANDO RESULTADOS FINAIS")
    print(f"{'='*80}")
    
    df_resultados_final = pd.DataFrame(resultados_resumo)
    
    # Salvar Excel
    df_resultados_final.to_excel(arquivo_final, index=False, engine='openpyxl')
    print(f"✓ Resultados salvos: {arquivo_final}")
    
    # Salvar CSV
    csv_final = arquivo_final.replace('.xlsx', '.csv')
    df_resultados_final.to_csv(csv_final, index=False)
    print(f"✓ Resultados salvos: {csv_final}")
    
    # =========================================================================
    # ESTATÍSTICAS GLOBAIS POR EMBARCAÇÃO
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"ESTATÍSTICAS GLOBAIS POR EMBARCAÇÃO")
    print(f"{'='*80}")
    
    for nome_emb, specs in embarcacoes_specs.items():
        coluna_taxa = f'Taxa_acerto_{nome_emb}_pct'
        
        print(f"\n  {nome_emb} ({specs['length']}m × {specs['width']}m):")
        print(f"    Taxa de acerto média: {df_resultados_final[coluna_taxa].mean():.2f}%")
        print(f"    Taxa mínima: {df_resultados_final[coluna_taxa].min():.2f}%")
        print(f"    Taxa máxima: {df_resultados_final[coluna_taxa].max():.2f}%")
        print(f"    Desvio padrão: {df_resultados_final[coluna_taxa].std():.2f}%")
    
    print(f"\n  CEP (independente de embarcação):")
    print(f"    CEP50 médio: {df_resultados_final['CEP50_m'].mean():.2f}m")
    print(f"    CEP90 médio: {df_resultados_final['CEP90_m'].mean():.2f}m")
    
    # =========================================================================
    # GRÁFICOS DE RESUMO
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"GERANDO GRÁFICOS DE RESUMO")
    print(f"{'='*80}")
    
    # Figura 1: Comparação de taxas de acerto por embarcação
    fig1, axes1 = plt.subplots(2, 3, figsize=(20, 12))
    axes1 = axes1.flatten()
    
    for idx, (nome_emb, specs) in enumerate(embarcacoes_specs.items()):
        ax = axes1[idx]
        coluna_taxa = f'Taxa_acerto_{nome_emb}_pct'
        
        ax.plot(df_resultados_final['Elevacao_deg'], df_resultados_final[coluna_taxa],
               'b-', linewidth=2, marker='o', markersize=2)
        ax.set_xlabel('Elevação [°]', fontsize=10)
        ax.set_ylabel('Taxa de Acerto [%]', fontsize=10)
        ax.set_title(f'{nome_emb}\n{specs["length"]}m × {specs["width"]}m', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        ax.set_ylim([0, 100])
    
    plt.suptitle(f'Taxa de Acerto por Embarcação vs Elevação\n' +
                 f'α=0°, β=0° | ΔElev σ=0.3°, ΔAzi σ=0.1° | {n_pontos_total} pontos | {n_simulacoes_total:,} simulações',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mc_alpha0_beta0_elev03_azi01_taxas_por_embarcacao.png', dpi=200, bbox_inches='tight')
    print(f"✓ Gráfico 1 salvo: mc_alpha0_beta0_elev03_azi01_taxas_por_embarcacao.png")
    
    # Figura 2: Comparação direta entre embarcações
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Todas as taxas vs Elevação
    ax1 = axes2[0, 0]
    for nome_emb, specs in embarcacoes_specs.items():
        coluna_taxa = f'Taxa_acerto_{nome_emb}_pct'
        ax1.plot(df_resultados_final['Elevacao_deg'], df_resultados_final[coluna_taxa],
                linewidth=2, marker='o', markersize=2, label=nome_emb, alpha=0.8)
    ax1.set_xlabel('Elevação [°]', fontsize=12)
    ax1.set_ylabel('Taxa de Acerto [%]', fontsize=12)
    ax1.set_title('Comparação: Todas as Embarcações', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.axhline(y=50, color='k', linestyle='--', alpha=0.3)
    
    # Gráfico 2: Todas as taxas vs Alcance
    ax2 = axes2[0, 1]
    for nome_emb, specs in embarcacoes_specs.items():
        coluna_taxa = f'Taxa_acerto_{nome_emb}_pct'
        ax2.plot(df_resultados_final['Alcance_m']/1000, df_resultados_final[coluna_taxa],
                linewidth=2, marker='o', markersize=2, label=nome_emb, alpha=0.8)
    ax2.set_xlabel('Alcance [km]', fontsize=12)
    ax2.set_ylabel('Taxa de Acerto [%]', fontsize=12)
    ax2.set_title('Comparação: Todas as Embarcações vs Alcance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.axhline(y=50, color='k', linestyle='--', alpha=0.3)
    
    # Gráfico 3: CEP50 vs Elevação
    ax3 = axes2[1, 0]
    ax3.plot(df_resultados_final['Elevacao_deg'], df_resultados_final['CEP50_m'],
            'g-', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Elevação [°]', fontsize=12)
    ax3.set_ylabel('CEP50 [m]', fontsize=12)
    ax3.set_title('CEP50 vs Elevação', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Taxa média por tamanho de embarcação
    ax4 = axes2[1, 1]
    nomes = []
    taxas_medias = []
    tamanhos = []
    for nome_emb, specs in embarcacoes_specs.items():
        coluna_taxa = f'Taxa_acerto_{nome_emb}_pct'
        nomes.append(nome_emb)
        taxas_medias.append(df_resultados_final[coluna_taxa].mean())
        tamanhos.append(specs['length'] * specs['width'])  # Área
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(nomes)))
    bars = ax4.bar(range(len(nomes)), taxas_medias, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(len(nomes)))
    ax4.set_xticklabels(nomes, rotation=45, ha='right')
    ax4.set_ylabel('Taxa de Acerto Média [%]', fontsize=12)
    ax4.set_title('Taxa Média por Embarcação', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    
    # Adicionar valores nas barras
    for bar, taxa in zip(bars, taxas_medias):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{taxa:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Simulação Monte Carlo - Múltiplas Embarcações\n' +
                 f'α=0°, β=0° | ΔElev σ=0.3°, ΔAzi σ=0.1° | {n_pontos_total} pontos | {n_simulacoes_total:,} simulações | ' +
                 f'Tempo: {tempo_total_final/60:.1f} min',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mc_alpha0_beta0_elev03_azi01_comparacao_geral.png', dpi=200, bbox_inches='tight')
    print(f"✓ Gráfico 2 salvo: mc_alpha0_beta0_elev03_azi01_comparacao_geral.png")
    
    plt.show()
    
    print(f"\n{'='*80}")
    print(f"SIMULAÇÃO MONTE CARLO MASSIVA CONCLUÍDA COM SUCESSO!")
    print(f"{'='*80}")
    print(f"\nArquivos gerados:")
    print(f"  1. {arquivo_final}")
    print(f"  2. {csv_final}")
    print(f"  3. {arquivo_backup}")
    print(f"  4. mc_alpha0_beta0_elev03_azi01_taxas_por_embarcacao.png")
    print(f"  5. mc_alpha0_beta0_elev03_azi01_comparacao_geral.png")
    print(f"\n{'='*80}")