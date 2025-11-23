# =============================================================================
# EXEMPLO DE USO - VARREDURA DE ÂNGULOS E POSICIONAMENTO DE ALVO
# =============================================================================

if __name__ == "__main__":
    
    import time  # Para medir tempo de execução
    
    print("="*80)
    print("SIMULADOR BALÍSTICO 6-DOF - VARREDURA DE ELEVAÇÃO E AZIMUTE")
    print("="*80)
    
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
    print(projectile.get_info())
    
    # =========================================================================
    # CONFIGURAÇÃO DA ARMA (FIXA)
    # =========================================================================
    print("\n" + "="*80)
    print("CONFIGURAÇÃO: VARREDURA DE ÂNGULOS DE ELEVAÇÃO")
    print("="*80)
    
    # Criar arma em terra (parâmetros fixos)
    weapon = Weapon(
        name="Canhão Naval 5\"/38",
        position=(0.0, 10.0, 0.0),  # Altura fixa de 10 metros
        elevation_deg=45.0,          # Será variado no loop
        azimuth_deg=-1.65,           # Azimute fixo em -1.65°
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
    # VARREDURA DE ÂNGULOS DE ELEVAÇÃO E AZIMUTE
    # =========================================================================
    print("\n" + "="*80)
    print("INICIANDO VARREDURA DE ELEVAÇÃO E AZIMUTE")
    print("="*80)
    print(f"  Elevação: 45.0° até -15.0° (passo: -0.1°)")
    print(f"  Azimute: -1.65° até 0.0° (passo: 0.05°)")
    print(f"  Altura inicial (fixa): 10.0 m")
    
    n_elevacoes = int((45.0 - (-15.0)) / 0.1) + 1
    n_azimutes = int((0.0 - (-1.65)) / 0.05) + 1
    total_simulacoes = n_elevacoes * n_azimutes
    
    print(f"  Total de simulações: {total_simulacoes:,} ({n_elevacoes} elev × {n_azimutes} azim)")
    print("="*80)
    
    # Arrays para armazenar resultados
    elevacoes = []
    azimutes = []
    alcances_x = []
    desvios_z = []
    alturas_max = []
    tempos_voo = []
    
    # Loop duplo de varredura
    contador = 0
    azimuth_atual = -1.65
    
    import time
    tempo_inicio = time.time()
    
    while azimuth_atual <= 0.0:
        print(f"\n--- Azimute: {azimuth_atual:.2f}° ---")
        
        elevation_atual = 45.0
        
        while elevation_atual >= -15.0:
            contador += 1
            
            # Atualizar ângulos
            weapon.set_firing_angles(elevation_deg=elevation_atual, azimuth_deg=azimuth_atual)
            
            # Simular (sem prints verbosos)
            result = simulator.simulate(
                max_time=100.0,
                alpha0_deg=0.0,
                beta0_deg=0.0,
                w_j0=5.0,
                w_k0=5.0
            )
            
            # Armazenar resultados
            elevacoes.append(elevation_atual)
            azimutes.append(azimuth_atual)
            alcances_x.append(result.alcance_max)
            desvios_z.append(result.z[-1])  # Desvio lateral final
            alturas_max.append(result.altura_max)
            tempos_voo.append(result.tempo_voo)
            
            # Print de progresso a cada 100 simulações
            if contador % 100 == 0 or contador == total_simulacoes:
                tempo_decorrido = time.time() - tempo_inicio
                tempo_por_sim = tempo_decorrido / contador
                tempo_restante = tempo_por_sim * (total_simulacoes - contador)
                
                print(f"  [{contador}/{total_simulacoes}] Elev: {elevation_atual:5.1f}° | Azim: {azimuth_atual:5.2f}° | "
                      f"Alcance: {result.alcance_max/1000:6.2f} km | Tempo restante: {tempo_restante/60:.1f} min")
            
            # Decrementar elevação
            elevation_atual -= 0.1
            elevation_atual = round(elevation_atual, 1)
        
        # Incrementar azimute
        azimuth_atual += 0.05
        azimuth_atual = round(azimuth_atual, 2)
    
    # Converter para arrays numpy
    elevacoes = np.array(elevacoes)
    azimutes = np.array(azimutes)
    alcances_x = np.array(alcances_x)
    desvios_z = np.array(desvios_z)
    alturas_max = np.array(alturas_max)
    tempos_voo = np.array(tempos_voo)
    
    tempo_total = time.time() - tempo_inicio
    print(f"\n? Varredura concluída em {tempo_total/60:.1f} minutos!")
    
    # =========================================================================
    # ENCONTRAR TIRO DE MAIOR ALCANCE
    # =========================================================================
    idx_max_alcance = np.argmax(alcances_x)
    elevacao_max_alcance = elevacoes[idx_max_alcance]
    azimute_max_alcance = azimutes[idx_max_alcance]
    x_max = alcances_x[idx_max_alcance]
    z_max = desvios_z[idx_max_alcance]
    
    print("\n" + "="*80)
    print("RESULTADO: TIRO DE MAIOR ALCANCE")
    print("="*80)
    print(f"  Elevação: {elevacao_max_alcance:.1f}°")
    print(f"  Azimute: {azimute_max_alcance:.2f}°")
    print(f"  Alcance (x): {x_max/1000:.3f} km = {x_max:.1f} m")
    print(f"  Desvio lateral (z): {z_max:.2f} m")
    print(f"  Altura máxima: {alturas_max[idx_max_alcance]:.2f} m")
    print(f"  Tempo de voo: {tempos_voo[idx_max_alcance]:.2f} s")
    
    # =========================================================================
    # CRIAR DRONE NAVAL "SEA BABY" NA POSIÇÃO DE MAIOR ALCANCE
    # =========================================================================
    print("\n" + "="*80)
    print("CRIANDO ALVO: DRONE NAVAL SEA BABY")
    print("="*80)
    
    # Criar drone naval na posição de impacto do tiro de maior alcance
    drone_sea_baby = Vessel(
        name="Drone Naval Sea Baby",
        center_position=(x_max, z_max),  # Posição (x, z) do tiro de maior alcance
        length=6.0,      # 6 metros de comprimento
        width=2.0,       # 2 metros de largura
        height=0.6,      # 0.6 metros de altura
        velocity=(0.0, 0.0)  # Parado
    )
    
    print(drone_sea_baby.get_info())
    print(f"\n  ?? Drone posicionado em: x={x_max:.1f} m, z={z_max:.2f} m")
    
    # Verificar se o tiro de maior alcance acerta o drone
    posicao_impacto = np.array([x_max, 0.0, z_max])  # y=0 (solo)
    acertou = drone_sea_baby.check_impact(posicao_impacto, time=0.0)
    
    if acertou:
        print(f"  ? ACERTO! O tiro de elevação {elevacao_max_alcance:.1f}° impacta o drone!")
    else:
        print(f"  ? Erro: O tiro não acerta o drone (verificar geometria)")
    
    # =========================================================================
    # SALVAR DADOS EM ARQUIVO
    # =========================================================================
    print("\n" + "="*80)
    print("SALVANDO DADOS")
    print("="*80)
    
    # Criar DataFrame com todos os resultados
    df_resultados = pd.DataFrame({
        'Elevacao_deg': elevacoes,
        'Azimute_deg': azimutes,
        'Alcance_x_m': alcances_x,
        'Desvio_z_m': desvios_z,
        'Altura_max_m': alturas_max,
        'Tempo_voo_s': tempos_voo
    })
    
    # Salvar em CSV
    df_resultados.to_csv('resultados_varredura_completa.csv', index=False)
    print(f"? Dados completos salvos em: resultados_varredura_completa.csv ({len(df_resultados)} simulações)")
    
    # =========================================================================
    # ANÁLISE: AZIMUTE ÓTIMO PARA DERIVA ZERO (POR ELEVAÇÃO)
    # =========================================================================
    print("\n" + "="*80)
    print("ANÁLISE: AZIMUTE ÓTIMO PARA DERIVA ~0 (POR ELEVAÇÃO)")
    print("="*80)
    
    elevacoes_unicas = np.unique(elevacoes)
    
    # Arrays para armazenar azimutes ótimos
    azimutes_otimos = []
    desvios_z_minimos = []
    alcances_correspondentes = []
    alturas_correspondentes = []
    tempos_correspondentes = []
    
    for elev in elevacoes_unicas:
        # Filtrar dados para esta elevação
        mask = np.abs(elevacoes - elev) < 0.001
        azimutes_elev = azimutes[mask]
        desvios_elev = desvios_z[mask]
        alcances_elev = alcances_x[mask]
        alturas_elev = alturas_max[mask]
        tempos_elev = tempos_voo[mask]
        
        # Encontrar azimute que minimiza |desvio_z|
        idx_min_desvio = np.argmin(np.abs(desvios_elev))
        
        azimute_otimo = azimutes_elev[idx_min_desvio]
        desvio_z_minimo = desvios_elev[idx_min_desvio]
        alcance_corresp = alcances_elev[idx_min_desvio]
        altura_corresp = alturas_elev[idx_min_desvio]
        tempo_corresp = tempos_elev[idx_min_desvio]
        
        azimutes_otimos.append(azimute_otimo)
        desvios_z_minimos.append(desvio_z_minimo)
        alcances_correspondentes.append(alcance_corresp)
        alturas_correspondentes.append(altura_corresp)
        tempos_correspondentes.append(tempo_corresp)
    
    # Criar DataFrame com azimutes ótimos
    df_azimutes_otimos = pd.DataFrame({
        'Elevacao_deg': elevacoes_unicas,
        'Azimute_otimo_deg': azimutes_otimos,
        'Desvio_z_resultante_m': desvios_z_minimos,
        'Alcance_x_m': alcances_correspondentes,
        'Altura_max_m': alturas_correspondentes,
        'Tempo_voo_s': tempos_correspondentes
    })
    
    # Salvar em Excel e CSV
    excel_path = 'azimutes_otimos_deriva_zero.xlsx'
    csv_path = 'azimutes_otimos_deriva_zero.csv'
    
    df_azimutes_otimos.to_excel(excel_path, index=False, engine='openpyxl')
    df_azimutes_otimos.to_csv(csv_path, index=False)
    
    print(f"? Azimutes ótimos salvos em:")
    print(f"    • Excel: {excel_path}")
    print(f"    • CSV:   {csv_path}")
    print(f"    Total de elevações analisadas: {len(elevacoes_unicas)}")
    
    # Estatísticas dos azimutes ótimos
    print(f"\n  ESTATÍSTICAS DOS AZIMUTES ÓTIMOS:")
    print(f"    Azimute médio: {np.mean(azimutes_otimos):.3f}°")
    print(f"    Azimute mínimo: {np.min(azimutes_otimos):.3f}°")
    print(f"    Azimute máximo: {np.max(azimutes_otimos):.3f}°")
    print(f"    Desvio Z médio resultante: {np.mean(np.abs(desvios_z_minimos)):.2f} m")
    print(f"    Desvio Z máximo resultante: {np.max(np.abs(desvios_z_minimos)):.2f} m")
    
    # Mostrar alguns exemplos
    print(f"\n  EXEMPLOS (primeiras 10 elevações):")
    print(f"  {'Elevação':>10s} | {'Azimute Ótimo':>15s} | {'Desvio Z':>12s} | {'Alcance':>10s}")
    print(f"  {'-'*10}-+-{'-'*15}-+-{'-'*12}-+-{'-'*10}")
    for i in range(min(10, len(elevacoes_unicas))):
        print(f"  {elevacoes_unicas[i]:10.1f}° | {azimutes_otimos[i]:15.3f}° | "
              f"{desvios_z_minimos[i]:11.2f} m | {alcances_correspondentes[i]/1000:9.2f} km")
    
    # =========================================================================
    # GRÁFICOS DE ANÁLISE
    # =========================================================================
    print("\nGerando gráficos de análise...")
    
    # Preparar dados para heatmaps (reshape para matriz)
    elevacoes_unicas_plot = np.unique(elevacoes)
    azimutes_unicos = np.unique(azimutes)
    
    # Criar matrizes 2D para heatmaps
    alcance_matrix = np.zeros((len(azimutes_unicos), len(elevacoes_unicas_plot)))
    desvio_z_matrix = np.zeros((len(azimutes_unicos), len(elevacoes_unicas_plot)))
    altura_matrix = np.zeros((len(azimutes_unicos), len(elevacoes_unicas_plot)))
    
    for i, dados in enumerate(zip(elevacoes, azimutes, alcances_x, desvios_z, alturas_max)):
        elev, azim, alc, desv, alt = dados
        i_elev = np.where(elevacoes_unicas_plot == elev)[0][0]
        i_azim = np.where(azimutes_unicos == azim)[0][0]
        alcance_matrix[i_azim, i_elev] = alc / 1000  # em km
        desvio_z_matrix[i_azim, i_elev] = desv
        altura_matrix[i_azim, i_elev] = alt
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(20, 14))
    
    # =========================================================================
    # Gráfico 1: Heatmap de Alcance vs Elevação e Azimute
    # =========================================================================
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.contourf(elevacoes_unicas_plot, azimutes_unicos, alcance_matrix, 
                       levels=30, cmap='viridis')
    ax1.plot(elevacao_max_alcance, azimute_max_alcance, 'r*', markersize=20, 
            markeredgecolor='white', markeredgewidth=2, label='Máximo')
    # Plotar curva de azimutes ótimos
    ax1.plot(elevacoes_unicas, azimutes_otimos, 'w-', linewidth=2, 
            linestyle='--', label='Azim. ótimo (deriva˜0)', alpha=0.8)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Alcance [km]', fontsize=10)
    ax1.set_xlabel('Elevação [°]', fontsize=11)
    ax1.set_ylabel('Azimute [°]', fontsize=11)
    ax1.set_title('Alcance vs Elevação e Azimute', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Gráfico 2: Heatmap de Desvio Lateral vs Elevação e Azimute
    # =========================================================================
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.contourf(elevacoes_unicas_plot, azimutes_unicos, desvio_z_matrix, 
                       levels=30, cmap='RdBu_r')
    ax2.plot(elevacao_max_alcance, azimute_max_alcance, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2)
    # Plotar curva de azimutes ótimos
    ax2.plot(elevacoes_unicas, azimutes_otimos, 'k-', linewidth=3, 
            linestyle='--', label='Azim. ótimo (deriva˜0)', alpha=0.9)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Desvio Z [m]', fontsize=10)
    ax2.set_xlabel('Elevação [°]', fontsize=11)
    ax2.set_ylabel('Azimute [°]', fontsize=11)
    ax2.set_title('Desvio Lateral vs Elevação e Azimute\n(Linha preta = deriva ˜ 0)', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Gráfico 3: NOVO - Azimute Ótimo vs Elevação
    # =========================================================================
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(elevacoes_unicas, azimutes_otimos, 'b-', linewidth=2.5, label='Azimute ótimo')
    ax3.fill_between(elevacoes_unicas, azimutes_otimos, azimutes_unicos.min(), 
                     alpha=0.2, color='blue')
    ax3.scatter(elevacoes_unicas[::10], azimutes_otimos[::10], c='red', s=30, zorder=5)
    ax3.set_xlabel('Elevação [°]', fontsize=11)
    ax3.set_ylabel('Azimute Ótimo [°]', fontsize=11)
    ax3.set_title('Azimute Ótimo para Deriva ˜ 0\nvs Elevação', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax3.axvline(x=0, color='k', linestyle=':', linewidth=1)
    
    # =========================================================================
    # Gráfico 4: Heatmap de Altura Máxima vs Elevação e Azimute
    # =========================================================================
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.contourf(elevacoes_unicas_plot, azimutes_unicos, altura_matrix, 
                       levels=30, cmap='plasma')
    ax4.plot(elevacao_max_alcance, azimute_max_alcance, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=2)
    ax4.plot(elevacoes_unicas, azimutes_otimos, 'w-', linewidth=2, 
            linestyle='--', alpha=0.8)
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('Altura Máxima [m]', fontsize=10)
    ax4.set_xlabel('Elevação [°]', fontsize=11)
    ax4.set_ylabel('Azimute [°]', fontsize=11)
    ax4.set_title('Altura Máxima vs Elevação e Azimute', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # Gráfico 5: Vista de Cima - Pontos de Impacto por Azimute
    # =========================================================================
    ax5 = plt.subplot(3, 3, 5)
    scatter = ax5.scatter(alcances_x/1000, desvios_z, c=azimutes, 
                         cmap='coolwarm', s=1, alpha=0.5)
    ax5.scatter([x_max/1000], [z_max], c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, label='Maior alcance', zorder=10)
    
    # Plotar pontos de azimutes ótimos (deriva ~0)
    for i in range(0, len(elevacoes_unicas), 10):  # Plotar alguns pontos
        elev = elevacoes_unicas[i]
        azim_opt = azimutes_otimos[i]
        # Encontrar o ponto correspondente
        mask = (np.abs(elevacoes - elev) < 0.001) & (np.abs(azimutes - azim_opt) < 0.001)
        if np.any(mask):
            ax5.scatter(alcances_x[mask]/1000, desvios_z[mask], 
                       c='lime', s=40, marker='o', edgecolors='black', 
                       linewidth=0.5, zorder=9, alpha=0.7)
    
    # Desenhar drone
    drone_x = x_max / 1000  # km
    drone_z = z_max
    drone_half_length = drone_sea_baby.length / 2
    drone_half_width = drone_sea_baby.width / 2
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((drone_x - drone_half_length/1000, drone_z - drone_half_width),
                     drone_sea_baby.length/1000, drone_sea_baby.width,
                     linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, 
                     label='Drone Sea Baby')
    ax5.add_patch(rect)
    
    # Linha Z=0
    ax5.axhline(y=0, color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label='Deriva = 0')
    
    cbar5 = plt.colorbar(scatter, ax=ax5)
    cbar5.set_label('Azimute [°]', fontsize=10)
    ax5.set_xlabel('Alcance X [km]', fontsize=11)
    ax5.set_ylabel('Desvio Lateral Z [m]', fontsize=11)
    ax5.set_title('Vista de Cima: Impactos + Drone\n(Verde = azim. ótimo)', 
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # =========================================================================
    # Gráfico 6: Alcance vs Elevação para Diferentes Azimutes
    # =========================================================================
    ax6 = plt.subplot(3, 3, 6)
    # Plotar curvas para alguns azimutes selecionados
    azimutes_plot = [-1.65, -1.5, -1.0, -0.5, 0.0]
    cores_azim = ['darkblue', 'blue', 'green', 'orange', 'red']
    
    for azim, cor in zip(azimutes_plot, cores_azim):
        mask = np.abs(azimutes - azim) < 0.001  # Máscara para esse azimute
        if np.any(mask):
            elev_azim = elevacoes[mask]
            alcance_azim = alcances_x[mask]
            ax6.plot(elev_azim, alcance_azim/1000, color=cor, linewidth=2, 
                    label=f'Azim={azim:.2f}°', alpha=0.8)
    
    # Plotar curva de azimutes ótimos
    ax6.plot(elevacoes_unicas, np.array(alcances_correspondentes)/1000, 
            'k--', linewidth=3, label='Azim. ótimo (deriva˜0)', alpha=0.9)
    
    ax6.scatter([elevacao_max_alcance], [x_max/1000], c='red', s=200, 
               marker='*', edgecolors='black', linewidth=2, zorder=10, 
               label='Máximo global')
    ax6.set_xlabel('Elevação [°]', fontsize=11)
    ax6.set_ylabel('Alcance [km]', fontsize=11)
    ax6.set_title('Alcance vs Elevação\n(para diferentes Azimutes)', 
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.axvline(x=0, color='k', linestyle=':', linewidth=0.8)
    
    # =========================================================================
    # Gráfico 7: NOVO - Desvio Z Residual com Azimute Ótimo
    # =========================================================================
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(elevacoes_unicas, np.abs(desvios_z_minimos), 'g-', linewidth=2.5)
    ax7.fill_between(elevacoes_unicas, 0, np.abs(desvios_z_minimos), 
                     alpha=0.3, color='green')
    ax7.scatter(elevacoes_unicas[::10], np.abs(desvios_z_minimos)[::10], 
               c='red', s=30, zorder=5)
    ax7.set_xlabel('Elevação [°]', fontsize=11)
    ax7.set_ylabel('|Desvio Z| Residual [m]', fontsize=11)
    ax7.set_title('Desvio Lateral Residual\n(com Azimute Ótimo)', 
                  fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=1, color='orange', linestyle='--', linewidth=1, 
               alpha=0.7, label='±1m')
    ax7.legend()
    
    # =========================================================================
    # Gráfico 8: Vista 3D - Alcance vs Elevação e Azimute
    # =========================================================================
    ax8 = plt.subplot(3, 3, 8, projection='3d')
    
    # Criar meshgrid para surface plot
    elev_grid, azim_grid = np.meshgrid(elevacoes_unicas_plot, azimutes_unicos)
    
    surf = ax8.plot_surface(elev_grid, azim_grid, alcance_matrix, 
                           cmap='viridis', alpha=0.8, edgecolor='none')
    ax8.scatter([elevacao_max_alcance], [azimute_max_alcance], [x_max/1000], 
               c='red', s=200, marker='*', edgecolors='black', linewidth=2, 
               label='Máximo')
    
    # Plotar curva de azimutes ótimos em 3D
    alcances_opt_km = np.array(alcances_correspondentes) / 1000
    ax8.plot(elevacoes_unicas, azimutes_otimos, alcances_opt_km,
            'k-', linewidth=3, label='Azim. ótimo')
    
    ax8.set_xlabel('Elevação [°]', fontsize=10)
    ax8.set_ylabel('Azimute [°]', fontsize=10)
    ax8.set_zlabel('Alcance [km]', fontsize=10)
    ax8.set_title('Superfície 3D: Alcance\n(Linha preta = deriva ˜ 0)', 
                  fontsize=12, fontweight='bold')
    plt.colorbar(surf, ax=ax8, shrink=0.5)
    ax8.view_init(elev=25, azim=45)
    ax8.legend()
    
    # =========================================================================
    # Gráfico 9: NOVO - Comparação de Desvio Z para Azimutes Extremos
    # =========================================================================
    ax9 = plt.subplot(3, 3, 9)
    
    # Azimute mínimo
    mask_min = np.abs(azimutes - azimutes_unicos.min()) < 0.001
    ax9.plot(elevacoes[mask_min], desvios_z[mask_min], 'b-', 
            linewidth=2, label=f'Azim = {azimutes_unicos.min():.2f}°')
    
    # Azimute máximo  
    mask_max = np.abs(azimutes - azimutes_unicos.max()) < 0.001
    ax9.plot(elevacoes[mask_max], desvios_z[mask_max], 'r-', 
            linewidth=2, label=f'Azim = {azimutes_unicos.max():.2f}°')
    
    # Azimutes ótimos
    ax9.plot(elevacoes_unicas, desvios_z_minimos, 'g-', 
            linewidth=3, label='Azim. ótimo', alpha=0.8)
    
    ax9.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax9.set_xlabel('Elevação [°]', fontsize=11)
    ax9.set_ylabel('Desvio Lateral Z [m]', fontsize=11)
    ax9.set_title('Comparação de Desvio Z\npara Diferentes Azimutes', 
                  fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Análise de Varredura Completa: Elevação (-15° a 45°) × Azimute (-1.65° a 0°)\n' + 
                 f'Altura: 10 m | Máximo em: Elev={elevacao_max_alcance:.1f}°, Azim={azimute_max_alcance:.2f}° ? {x_max:.0f} m\n' +
                 f'Linha preta/verde = Azimutes ótimos para deriva ˜ 0',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analise_varredura_completa.png', dpi=150, bbox_inches='tight')
    print(f"? Gráficos salvos em: analise_varredura_completa.png")
    
    # =========================================================================
    # GRÁFICO ADICIONAL: Diagrama de dispersão detalhado com drone
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(alcances_x/1000, desvios_z, c=elevacoes, 
                        cmap='jet', s=5, alpha=0.6)
    ax.scatter([x_max/1000], [z_max], c='red', s=300, marker='*', 
              edgecolors='black', linewidth=2, label='Tiro de maior alcance', zorder=10)
    
    # Desenhar drone
    rect2 = Rectangle((drone_x - drone_half_length/1000, drone_z - drone_half_width),
                      drone_sea_baby.length/1000, drone_sea_baby.width,
                      linewidth=3, edgecolor='red', facecolor='red', alpha=0.4,
                      label='Drone Sea Baby (6×2×0.6 m)')
    ax.add_patch(rect2)
    
    # Adicionar texto com informações
    ax.text(0.02, 0.98, f'Total de tiros: {len(alcances_x):,}\n' + 
                        f'Elevações: -15° a 45°\n' +
                        f'Azimutes: -1.65° a 0°\n' +
                        f'Melhor tiro:\n' +
                        f'  Elev: {elevacao_max_alcance:.1f}°\n' +
                        f'  Azim: {azimute_max_alcance:.2f}°\n' +
                        f'  Alcance: {x_max:.0f} m\n' +
                        f'  Desvio Z: {z_max:.1f} m',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Elevação [°]', fontsize=11)
    ax.set_xlabel('Alcance X [km]', fontsize=12)
    ax.set_ylabel('Desvio Lateral Z [m]', fontsize=12)
    ax.set_title('Diagrama de Dispersão Completo: Todos os Pontos de Impacto + Alvo', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagrama_dispersao_completo.png', dpi=150, bbox_inches='tight')
    print(f"? Diagrama adicional salvo em: diagrama_dispersao_completo.png")
    
    plt.show()
    
    # =========================================================================
    # ESTATÍSTICAS FINAIS
    # =========================================================================
    print("\n" + "="*80)
    print("ESTATÍSTICAS FINAIS DA VARREDURA")
    print("="*80)
    print(f"  Total de simulações: {len(alcances_x):,}")
    print(f"  Tempo total: {tempo_total/60:.1f} minutos")
    print(f"  Tempo médio por simulação: {tempo_total/len(alcances_x):.2f} s")
    print(f"\n  ALCANCES:")
    print(f"    Máximo: {np.max(alcances_x)/1000:.3f} km")
    print(f"      ? Elevação: {elevacao_max_alcance:.1f}°, Azimute: {azimute_max_alcance:.2f}°")
    print(f"    Mínimo: {np.min(alcances_x)/1000:.3f} km")
    print(f"      ? Elevação: {elevacoes[np.argmin(alcances_x)]:.1f}°, Azimute: {azimutes[np.argmin(alcances_x)]:.2f}°")
    print(f"    Médio: {np.mean(alcances_x)/1000:.3f} km")
    print(f"\n  DESVIO LATERAL:")
    print(f"    Médio (todos os tiros): {np.mean(desvios_z):.2f} m")
    print(f"    Máximo (absoluto, todos os tiros): {np.max(np.abs(desvios_z)):.2f} m")
    print(f"    Médio (com azimute ótimo): {np.mean(np.abs(desvios_z_minimos)):.2f} m")
    print(f"    Máximo (com azimute ótimo): {np.max(np.abs(desvios_z_minimos)):.2f} m")
    print(f"\n  OUTRAS MÉTRICAS:")
    print(f"    Altura máxima alcançada: {np.max(alturas_max):.2f} m")
    print(f"    Tempo de voo médio: {np.mean(tempos_voo):.2f} s")
    
    print("\n" + "="*80)
    print("SIMULAÇÃO COMPLETA!")
    print("="*80)
    print(f"\nArquivos gerados:")
    print(f"  1. resultados_varredura_completa.csv ({len(alcances_x):,} linhas)")
    print(f"  2. azimutes_otimos_deriva_zero.xlsx ({len(elevacoes_unicas)} linhas)")
    print(f"  3. azimutes_otimos_deriva_zero.csv ({len(elevacoes_unicas)} linhas)")
    print(f"  4. analise_varredura_completa.png (9 painéis)")
    print(f"  5. diagrama_dispersao_completo.png")
    print(f"\n?? Drone naval Sea Baby posicionado em:")
    print(f"     X = {x_max:.1f} m, Z = {z_max:.2f} m")
    print(f"     (Tiro ótimo: Elevação={elevacao_max_alcance:.1f}°, Azimute={azimute_max_alcance:.2f}°)")
    print(f"\n?? Azimutes ótimos para deriva ˜ 0:")
    print(f"     Salvos em: azimutes_otimos_deriva_zero.xlsx")
    print(f"     Para cada elevação, mostra o azimute que minimiza |desvio_z|")