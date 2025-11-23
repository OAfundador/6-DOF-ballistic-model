# =============================================================================
# MAIN DE TESTE - TIRO SEM RANDOMIZAÇÃO EM 39.6°
# =============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("TESTE DE PLOTAGEM ESTILO 3BLUE1BROWN (FUNDO BRANCO)")
    print("Tiro único sem randomização - Elevação 39.6°")
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
    
    # Criar arma em terra
    weapon = Weapon(
        name="Canhão Naval 5\"/38",
        position=(0.0, 10.0, 0.0),
        elevation_deg=43.3,
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
    
    # Executar simulação
    print("\nExecutando simulação...")
    result = simulator.simulate(
        max_time=100.0,
        alpha0_deg=0.0,
        beta0_deg=0.0,
        w_j0=5.0,
        w_k0=5.0,
        rtol=1e-7,
        atol=1e-8
    )
    
    # Imprimir estatísticas
    result.print_statistics()
    
    # Gerar todos os gráficos
    result.plot_all_graphs()
    
    print("\n" + "="*80)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("="*80)
