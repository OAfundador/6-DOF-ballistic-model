# =============================================================================
# SIMULADOR BALÍSTICO 6-DOF
# =============================================================================

import numpy as np
from math import sqrt, pi, sin, cos, atan2, acos
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import pandas as pd
from scipy.interpolate import interp1d, RectBivariateSpline
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CLASSE PARA INTERPOLAÇÃO DOS COEFICIENTES AERODINÂMICOS
# =============================================================================

class RealAerodynamicCoefficients:
    """
    Classe para carregar e interpolar coeficientes aerodinâmicos.
    Pré-Computa um grid 2D (Mach x alpha) para rodar mais rapidamente.
    """
    
    def __init__(self, csv_path='C:\\Users\\DELL\\Downloads\\Coeficientes que vi 2 casas.csv'):
        """
        Carrega os coeficientes e pré-computa grid 2D.
        """
        print("\n" + "="*60)
        print("CARREGANDO COEFICIENTES AERODINÂMICOS (GRID 2D)")
        print("="*60)
        
        # Carregar arquivo Excel
        self.df = pd.read_excel(csv_path)
        print(f"✓ Arquivo carregado: {len(self.df)} pontos de Mach")
        print(f"  Faixa de Mach: {self.df['Match'].min():.2f} a {self.df['Match'].max():.2f}") # Observe que a tabela de coeficientes está esrcito "Match" errôneamente
        
        # Arrays de Mach da tabela
        mach_values = self.df['Match'].values
        self.mach_min = float(mach_values.min())
        self.mach_max = float(mach_values.max())
        
        # Lista de coeficientes (Detalhes encontrados no relatório)
        coeff_names = ['CX0', 'CX2', 'CNA', 'CMA', 'CPN', 'CYP', 
                      'CNPA', 'CNPA3', 'CNPA5', 'CPF1', 'CPF5', 
                      'CNPA-5', 'CMQ', 'CLP']
        
        # Pré-Computar grid 2D
        print("\nPré-computando grid 2D (Mach x Alpha)...")
        
        # Grid de Mach
        self.mach_grid = np.linspace(self.mach_min, self.mach_max, 100)
        
        # Grid de alpha: -10 a +10 graus (Escolha para otimizar, funciona adequadamente na faixa que estudaremos, possível problema para disparos semi-vertiais)
        self.alpha_grid = np.linspace(-np.radians(10), np.radians(10), 100)
        
        # Criar interpoladores 1D para cada coeficiente
        self.splines_1d = {}
        for col in coeff_names:
            if col in self.df.columns:
                self.splines_1d[col] = interp1d(
                    mach_values, 
                    self.df[col].values,
                    kind='cubic',
                    bounds_error=False,
                    fill_value=(self.df[col].values[0], self.df[col].values[-1])
                )
        
        # Pré-Computar todos os coeficientes base na grid de Mach
        self.grid_2d = {}
        for name, spline in self.splines_1d.items():
            self.grid_2d[name] = spline(self.mach_grid)
        
        # Pré-Computar coeficientes que dependem de alpha
        mach_mesh, alpha_mesh = np.meshgrid(self.mach_grid, self.alpha_grid, indexing='ij')
        
        print(f"  Grid: {len(self.mach_grid)} Machs x {len(self.alpha_grid)} alphas")
        print(f"  Range de alpha: [{np.degrees(self.alpha_grid[0]):.1f}°, {np.degrees(self.alpha_grid[-1]):.1f}°]")
        
        # Interpolar coeficientes base para a grid
        CX0_grid = np.zeros_like(mach_mesh)
        CX2_grid = np.zeros_like(mach_mesh)
        CNA_grid = np.zeros_like(mach_mesh)
        CNPA_grid = np.zeros_like(mach_mesh)
        CNPA3_grid = np.zeros_like(mach_mesh)
        CNPA5_grid = np.zeros_like(mach_mesh)
        
        for i, m in enumerate(self.mach_grid):
            if 'CX0' in self.grid_2d:
                CX0_grid[i, :] = self.grid_2d['CX0'][i]
            if 'CX2' in self.grid_2d:
                CX2_grid[i, :] = self.grid_2d['CX2'][i]
            if 'CNA' in self.grid_2d:
                CNA_grid[i, :] = self.grid_2d['CNA'][i]
            if 'CNPA' in self.grid_2d:
                CNPA_grid[i, :] = self.grid_2d['CNPA'][i]
            if 'CNPA3' in self.grid_2d:
                CNPA3_grid[i, :] = self.grid_2d['CNPA3'][i]
            if 'CNPA5' in self.grid_2d:
                CNPA5_grid[i, :] = self.grid_2d['CNPA5'][i]
        
        # Pré-Computar CD_total e CNP_total
        sin_alpha_mesh = np.sin(alpha_mesh)
        cos_alpha_mesh = np.cos(alpha_mesh)
        sin_alpha_2_mesh = np.sin(alpha_mesh)**2
        sin_alpha_3_mesh = sin_alpha_mesh**3
        sin_alpha_5th_mesh = sin_alpha_mesh**5
        
        # CX total (para uso no CLA)
        CX_total = CX0_grid + CX2_grid * sin_alpha_2_mesh
        
        # CD_total com o termo de CNA
        self.grid_2d['CD_total'] = CX_total*cos_alpha_mesh - (CNA_grid*sin_alpha_2_mesh)
        
        # CLA = CNA*cos(alpha) - CX
        self.grid_2d['CLA_total'] = CNA_grid * cos_alpha_mesh - CX_total
        
        self.grid_2d['CNP_total'] = (CNPA_grid * np.sign(alpha_mesh) +
                                      CNPA3_grid * sin_alpha_3_mesh+ 
                                      CNPA5_grid * sin_alpha_5th_mesh)
        
        
        # Criar interpoladores 2D
        self.interp_2d = {}
        self.interp_2d['CD_total'] = RectBivariateSpline(
            self.mach_grid, self.alpha_grid, self.grid_2d['CD_total'], kx=3, ky=3)
        self.interp_2d['CLA_total'] = RectBivariateSpline(
            self.mach_grid, self.alpha_grid, self.grid_2d['CLA_total'], kx=3, ky=3)
        self.interp_2d['CNP_total'] = RectBivariateSpline(
            self.mach_grid, self.alpha_grid, self.grid_2d['CNP_total'], kx=3, ky=3)
        
        # Coeficientes que não dependem 
        for name in ['CNA', 'CMA', 'CMQ', 'CLP', 'CYP']:
            if name in self.grid_2d:
                self.interp_2d[name] = interp1d(
                    self.mach_grid,
                    self.grid_2d[name],
                    kind='cubic',
                    bounds_error=False,
                    fill_value=(self.grid_2d[name][0], self.grid_2d[name][-1])
                )
        
        print("✓ Grid pré-computada com interpoladores prontos!")
        
    def get_coefficients(self, mach, alpha_rad=0.0):
        """
        Utilização dos coeficientes pré-computados
        """
        mach = np.clip(mach, self.mach_min, self.mach_max)
        alpha_rad = np.clip(alpha_rad, self.alpha_grid[0], self.alpha_grid[-1])
        
        coeffs = {}
        
        # Coeficientes que necessitam de alpha
        coeffs['CD_total'] = float(self.interp_2d['CD_total'](mach, alpha_rad))
        coeffs['CNP_total'] = float(self.interp_2d['CNP_total'](mach, alpha_rad))
        coeffs['CLA_total'] = float(self.interp_2d['CLA_total'](mach, alpha_rad))
        
        # Coeficientes fixos
        for name in ['CNA', 'CMA', 'CMQ', 'CLP', 'CYP']:
            if name in self.interp_2d:
                coeffs[name] = float(self.interp_2d[name](mach))
        
        return coeffs


# =============================================================================
# CLASSE PROJÉTIL (MUNIÇÃO)
# =============================================================================

class Projectile:
    """
    Classe que representa um projétil naval com todas suas características físicas
    e balísticas.
    """
    
    def __init__(self, name="Projétil Naval", mass_kg=None, diameter_m=None,
                 I_P_kg_m2=None, I_T_kg_m2=None, rifling_twist_calibers=25.0):
        """
        Inicializa um projétil.
        
        Parâmetros:
        -----------
        name : str
            Nome/tipo do projétil
        mass_kg : float
            Massa em kg
        diameter_m : float
            Diâmetro em metros
        I_P_kg_m2 : float
            Momento de inércia polar em kg·m²
        I_T_kg_m2 : float
            Momento de inércia transversal em kg·m²
        rifling_twist_calibers : float
            Taxa de rifling em calibres por volta completa
        """
        self.name = name
        self.mass = mass_kg
        self.diameter = diameter_m
        self.I_P = I_P_kg_m2
        self.I_T = I_T_kg_m2
        self.rifling_twist = rifling_twist_calibers
        
        # Área de referência
        if diameter_m is not None:
            self.S = pi * (diameter_m / 2) ** 2
        else:
            self.S = None
    
    @classmethod
    def from_imperial(cls, name, mass_lb, diameter_in, I_P_lbin2, I_T_lbin2, 
                     rifling_twist_calibers=25.0):
        """
        Cria um projétil a partir de unidades imperiais.
        
        Parâmetros:
        -----------
        mass_lb : float
            Massa em libras
        diameter_in : float
            Diâmetro em polegadas
        I_P_lbin2 : float
            Momento de inércia polar em lb·in²
        I_T_lbin2 : float
            Momento de inércia transversal em lb·in²
        """
        LB_TO_KG = 0.453592
        IN_TO_M = 0.0254
        LBIN2_TO_KGM2 = LB_TO_KG * (IN_TO_M ** 2)
        
        mass_kg = mass_lb * LB_TO_KG
        diameter_m = diameter_in * IN_TO_M
        I_P_kg_m2 = I_P_lbin2 * LBIN2_TO_KGM2
        I_T_kg_m2 = I_T_lbin2 * LBIN2_TO_KGM2
        
        return cls(name, mass_kg, diameter_m, I_P_kg_m2, I_T_kg_m2, 
                  rifling_twist_calibers)
    
    def calculate_initial_spin(self, muzzle_velocity):
        """
        Calcula o spin inicial baseado no rifling do canhão.
        
        Parâmetros:
        -----------
        muzzle_velocity_mps : float
            Velocidade na boca do canhão em m/s
            
        Retorna:
        --------
        float : spin inicial em rad/s
        """
        n = self.rifling_twist
        p0 = (2 * np.pi * muzzle_velocity) / (n * self.diameter)
        return p0
    
    def get_info(self):
        """Retorna informações formatadas sobre o projétil."""
        info = f"\n{'='*60}\n"
        info += f"PROJÉTIL: {self.name}\n"
        info += f"{'='*60}\n"
        info += f"  Massa: {self.mass:.2f} kg\n"
        info += f"  Diâmetro: {self.diameter*1000:.1f} mm\n"
        info += f"  I_P: {self.I_P:.6f} kg·m²\n"
        info += f"  I_T: {self.I_T:.6f} kg·m²\n"
        info += f"  I_P/I_T: {self.I_P/self.I_T:.6f}\n"
        info += f"  Área de referência: {self.S:.6f} m²\n"
        info += f"  Rifling twist: {self.rifling_twist:.1f} calibres/volta\n"
        return info


# =============================================================================
# CLASSE ARMA
# =============================================================================

class Weapon:
    """
    Classe que representa uma arma (canhão) com suas características e posição.
    """
    
    def __init__(self, name="Canhão Naval", position=(0.0, 0.0, 0.0),
                 elevation_deg=45.0, azimuth_deg=0.0, rate_of_fire_rpm=15.0,
                 muzzle_velocity_mps=807.0, mounted_on_vessel=None):
        """
        Inicializa uma arma.
        
        Parâmetros:
        -----------
        name : str
            Nome/tipo da arma
        position : tuple (x, y, z)
            Posição da arma em metros RELATIVA à embarcação (x=frente, y=altura, z=direita)
        elevation_deg : float
            Elevação em graus
        azimuth_deg : float
            Azimute em graus
        rate_of_fire_rpm : float
            Taxa de tiro em tiros por minuto
        muzzle_velocity_mps : float
            Velocidade na boca em m/s
        mounted_on_vessel : Vessel, optional
            Embarcação na qual a arma está montada (None = arma em terra)
        """
        self.name = name
        self.position = np.array(position, dtype=float)  # [x, y, z] relativo à embarcação
        self.elevation = np.radians(elevation_deg)
        self.azimuth = np.radians(azimuth_deg)
        self.rate_of_fire = rate_of_fire_rpm
        self.muzzle_velocity = muzzle_velocity_mps
        self.mounted_on_vessel = mounted_on_vessel  # Referência à embarcação
    
    def set_firing_angles(self, elevation_deg, azimuth_deg):
        """Define os ângulos de tiro."""
        self.elevation = np.radians(elevation_deg)
        self.azimuth = np.radians(azimuth_deg)
    
    def get_absolute_position(self, time=0.0):
        """
        Retorna a posição absoluta da arma no espaço.
        Se montada em embarcação, considera movimento da embarcação.
        
        Parâmetros:
        -----------
        time : float
            Tempo em segundos (para calcular posição com movimento)
            
        Retorna:
        --------
        array : [x, y, z] posição absoluta em metros
        """
        if self.mounted_on_vessel is None:
            # Arma em terra - posição é absoluta
            return self.position.copy()
        else:
            # Arma montada em embarcação
            vessel_bounds = self.mounted_on_vessel.get_bounds(time)
            vessel_center_x = (vessel_bounds['x_min'] + vessel_bounds['x_max']) / 2
            vessel_center_z = (vessel_bounds['z_min'] + vessel_bounds['z_max']) / 2
            
            # Posição absoluta = posição do centro da embarcação + posição relativa da arma
            absolute_pos = np.array([
                vessel_center_x + self.position[0],
                self.position[1],  # Altura não muda
                vessel_center_z + self.position[2]
            ])
            return absolute_pos
    
    def get_velocity(self):
        """
        Retorna a velocidade da arma (velocidade da embarcação se montada).
        
        Retorna:
        --------
        array : [vx, vy, vz] velocidade em m/s
        """
        if self.mounted_on_vessel is None:
            # Arma em terra - velocidade zero
            return np.array([0.0, 0.0, 0.0])
        else:
            # Arma montada em embarcação - herda velocidade da embarcação
            return np.array([
                self.mounted_on_vessel.velocity[0],  # vx
                0.0,                                   # vy (embarcação não sobe/desce)
                self.mounted_on_vessel.velocity[1]   # vz
            ])
    
    def calculate_firing_angles(self):
        """
        Calcula os ângulos theta0 e phi0 conforme a convenção do simulador.
        
        Retorna:
        --------
        tuple : (theta0, phi0) em radianos
        """
        E = self.elevation # Elevação
        A = self.azimuth   # Azimutal
        
        theta0 = np.arcsin(np.cos(E) * np.sin(A))
        phi0 = np.arcsin(np.sin(E) / np.cos(theta0)) if np.cos(theta0) != 0 else np.pi/2
        
        return theta0, phi0
    
    def get_info(self):
        """Retorna informações formatadas sobre a arma."""
        info = f"\n{'='*60}\n"
        info += f"ARMA: {self.name}\n"
        info += f"{'='*60}\n"
        
        if self.mounted_on_vessel is None:
            info += f"  Posição (x, y, z): ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}) m\n"
            info += f"  Montada em: Terra (velocidade = 0)\n"
        else:
            abs_pos = self.get_absolute_position()
            vessel_vel = self.get_velocity()
            info += f"  Posição relativa (x, y, z): ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}) m\n"
            info += f"  Posição absoluta (x, y, z): ({abs_pos[0]:.1f}, {abs_pos[1]:.1f}, {abs_pos[2]:.1f}) m\n"
            info += f"  Montada em: {self.mounted_on_vessel.name}\n"
            info += f"  Velocidade da plataforma: ({vessel_vel[0]:.1f}, {vessel_vel[1]:.1f}, {vessel_vel[2]:.1f}) m/s\n"
        
        info += f"  Elevação: {np.degrees(self.elevation):.1f}°\n"
        info += f"  Azimute: {np.degrees(self.azimuth):.1f}°\n"
        info += f"  Taxa de tiro: {self.rate_of_fire:.1f} tiros/min\n"
        info += f"  Velocidade na boca: {self.muzzle_velocity:.1f} m/s\n"
        return info


# =============================================================================
# CLASSE EMBARCAÇÃO 
# =============================================================================

class Vessel:
    """
    Classe que representa uma embarcação como um paralelepípedo.
    Preparada para simulações futuras envolvendo alvos móveis.
    """
    
    def __init__(self, name="Embarcação", center_position=(0.0, 0.0),
                 length=100.0, width=20.0, height=30.0,
                 velocity=(0.0, 0.0)):
        """
        Inicializa uma embarcação.
        
        Parâmetros:
        -----------
        name : str
            Nome da embarcação
        center_position : tuple (x, z)
            Posição do centro da embarcação (x=frente, z=direita)
        length : float
            Comprimento da embarcação em metros (eixo x)
        width : float
            Largura da embarcação em metros (eixo z)
        height : float
            Altura do casco em metros (eixo y)
        velocity : tuple (vx, vz)
            Velocidade da embarcação em m/s
        """
        self.name = name
        self.center = np.array(center_position, dtype=float)  # [x, z]
        self.length = length  # Comprimento (eixo x)
        self.width = width    # Largura (eixo z)
        self.height = height  # Altura (eixo y)
        self.velocity = np.array(velocity, dtype=float)  # [vx, vz]
    
    def get_bounds(self, time=0.0):
        """
        Retorna os limites da embarcação no espaço 3D no instante dado.
        
        Parâmetros:
        -----------
        time : float
            Tempo em segundos (para calcular posição com movimento)
            
        Retorna:
        --------
        dict : limites em x, y, z
        """
        # Posição atual considerando movimento
        current_center = self.center + self.velocity * time
        
        bounds = {
            'x_min': current_center[0] - self.length / 2,
            'x_max': current_center[0] + self.length / 2,
            'y_min': 0.0,  # Nível do mar
            'y_max': self.height,
            'z_min': current_center[1] - self.width / 2,
            'z_max': current_center[1] + self.width / 2
        }
        return bounds
    
    def check_impact(self, projectile_position, time=0.0, check_height=True):
        """
        Verifica se um projétil impactou a embarcação.
        
        Parâmetros:
        -----------
        projectile_position : array-like [x, y, z]
            Posição do projétil
        time : float
            Tempo atual
        check_height : bool
            Se True, verifica altura Y. Se False, ignora Y (útil para alvos no solo)
            
        Retorna:
        --------
        bool : True se houve impacto
        """
        bounds = self.get_bounds(time)
        x, y, z = projectile_position
        
        if check_height:
            impact = (bounds['x_min'] <= x <= bounds['x_max'] and
                    bounds['y_min'] <= y <= bounds['y_max'] and
                    bounds['z_min'] <= z <= bounds['z_max'])
        else:
            # Ignorar altura Y (para alvos no solo)
            impact = (bounds['x_min'] <= x <= bounds['x_max'] and
                    bounds['z_min'] <= z <= bounds['z_max'])
        
        return impact
    
    def get_info(self):
        """Retorna informações formatadas sobre a embarcação."""
        info = f"\n{'='*60}\n"
        info += f"EMBARCAÇÃO: {self.name}\n"
        info += f"{'='*60}\n"
        info += f"  Posição do centro (x, z): ({self.center[0]:.1f}, {self.center[1]:.1f}) m\n"
        info += f"  Dimensões (L×W×H): {self.length:.1f} × {self.width:.1f} × {self.height:.1f} m\n"
        info += f"  Velocidade (vx, vz): ({self.velocity[0]:.1f}, {self.velocity[1]:.1f}) m/s\n"
        return info


# =============================================================================
# CLASSE AMBIENTE
# =============================================================================

@dataclass
class Environment:
    """
    Classe que define as condições ambientais da simulação.
    """
    rho: float = 1.225  # Densidade do ar [kg/m³]
    g: float = 9.81     # Aceleração da gravidade [m/s²]
    W1: float = 0.0     # Vento na direção x [m/s]
    W2: float = 0.0     # Vento na direção y [m/s]
    W3: float = 0.0     # Vento na direção z [m/s]
    sound_speed: float = 340.0  # Velocidade do som [m/s]


# =============================================================================
# CLASSE SIMULADOR BALÍSTICO
# =============================================================================

class BallisticSimulator:
    """
    Classe principal que gerencia a simulação balística 6-DOF.
    """
    
    def __init__(self, projectile, weapon, environment, aero_coeffs):
        """
        Inicializa o simulador.
        
        Parâmetros:
        -----------
        projectile : Projectile
            Objeto projétil
        weapon : Weapon
            Objeto arma
        environment : Environment
            Condições ambientais
        aero_coeffs : RealAerodynamicCoefficients
            Coeficientes aerodinâmicos
        """
        self.projectile = projectile
        self.weapon = weapon
        self.environment = environment
        self.aero_coeffs = aero_coeffs
        
        # Resultado da simulação
        self.result = None
    
    def build_initial_conditions(self, alpha0_deg=0.0, beta0_deg=0.0,
                                 w_j0=5.0, w_k0=5.0):
        """
        Constrói condições iniciais para a simulação.
        IMPORTANTE: Considera a velocidade da embarcação se a arma estiver montada em uma.
        
        Parâmetros:
        -----------
        alpha0_deg : float
            Ângulo de arfagem inicial em graus
        beta0_deg : float
            Ângulo de guinada inicial em graus
        w_j0 : float
            Velocidade angular em j' [rad/s]
        w_k0 : float
            Velocidade angular em k' [rad/s]
            
        Retorna:
        --------
        array : vetor de estado inicial [V1, V2, V3, h1, h2, h3, i1, i2, i3, x, y, z]
        """
        # Ângulos de tiro
        theta0, phi0 = self.weapon.calculate_firing_angles()
        alpha0 = np.radians(alpha0_deg)
        beta0 = np.radians(beta0_deg)
        
        # Velocidade inicial DO PROJÉTIL RELATIVA À ARMA (velocidade na boca)
        V0 = self.weapon.muzzle_velocity
        V1_rel = V0 * cos(theta0) * cos(phi0)
        V2_rel = V0 * cos(theta0) * sin(phi0)
        V3_rel = V0 * sin(theta0)
        
        # Velocidade da plataforma (embarcação)
        platform_velocity = self.weapon.get_velocity()
        
        # VELOCIDADE ABSOLUTA = Velocidade relativa + Velocidade da plataforma
        V1_0 = V1_rel + platform_velocity[0]
        V2_0 = V2_rel + platform_velocity[1]
        V3_0 = V3_rel + platform_velocity[2]
        
        # Spin inicial
        w_i0 = self.projectile.calculate_initial_spin(V0)
        
        '''
        print(f"\n1. Velocidade inicial:")
        if np.any(platform_velocity != 0):
            print(f"   Velocidade relativa (à arma): [{V1_rel:.2f}, {V2_rel:.2f}, {V3_rel:.2f}] m/s")
            print(f"   Velocidade da plataforma: [{platform_velocity[0]:.2f}, {platform_velocity[1]:.2f}, {platform_velocity[2]:.2f}] m/s")
            print(f"   Velocidade ABSOLUTA: [{V1_0:.2f}, {V2_0:.2f}, {V3_0:.2f}] m/s")
        else:
            print(f"   V = [{V1_0:.2f}, {V2_0:.2f}, {V3_0:.2f}] m/s (plataforma estacionária)")
        print(f"   |V| = {sqrt(V1_0**2 + V2_0**2 + V3_0**2):.2f} m/s")
        '''

        # Eixo polar i'
        phi_eff = phi0 + alpha0
        theta_eff = theta0 + beta0
        
        i1_0 = cos(phi_eff) * cos(theta_eff)
        i2_0 = cos(theta_eff) * sin(phi_eff)
        i3_0 = sin(theta_eff)
        
        '''
        print(f"\n2. Eixo polar i':")
        print(f"   φ_eff = {np.degrees(phi0):.2f}° + {np.degrees(alpha0):.2f}° = {np.degrees(phi_eff):.2f}°")
        print(f"   θ_eff = {np.degrees(theta0):.2f}° + {np.degrees(beta0):.2f}° = {np.degrees(theta_eff):.2f}°")
        print(f"   i' = [{i1_0:.6f}, {i2_0:.6f}, {i3_0:.6f}]")
        '''
        
        # Eixos j' e k'
        Q = sin(theta_eff)**2 + cos(theta_eff)**2 * cos(phi_eff)**2
        sqrt_Q = sqrt(Q)
        
        j1_0 = -(sin(phi_eff) * cos(phi_eff) * cos(theta_eff)**2) / sqrt_Q
        j2_0 = (cos(theta_eff)**2 * cos(phi_eff)**2 + sin(theta_eff)**2) / sqrt_Q
        j3_0 = -(sin(theta_eff) * cos(theta_eff) * sin(phi_eff)) / sqrt_Q

        k1_0 = -sin(theta_eff) / sqrt_Q
        k2_0 = 0.0
        k3_0 = (cos(phi_eff) * cos(theta_eff)) / sqrt_Q
        
        # di'/dt
        di1_dt = (w_j0 * sin(theta_eff) -
                  w_k0 * cos(theta_eff)**2 * sin(phi_eff) * cos(phi_eff)) / sqrt_Q
        
        di2_dt = (w_k0 / sqrt_Q) * (cos(theta_eff)**2 * cos(phi_eff)**2 + 
                                       sin(theta_eff)**2)
        
        di3_dt = (-w_j0 * cos(theta_eff) * cos(phi_eff)
                  - w_k0 * sin(phi_eff) * cos(theta_eff) * sin(theta_eff)) / sqrt_Q
        
        # Momento angular h
        omega1_inertial = w_i0
        
        I_P = self.projectile.I_P
        I_T = self.projectile.I_T
        
        term1_h1 = (I_P / I_T) * omega1_inertial * i1_0
        term1_h2 = (I_P / I_T) * omega1_inertial * i2_0
        term1_h3 = (I_P / I_T) * omega1_inertial * i3_0
        
        term2_h1 = i2_0 * di3_dt - i3_0 * di2_dt
        term2_h2 = i3_0 * di1_dt - i1_0 * di3_dt
        term2_h3 = i1_0 * di2_dt - i2_0 * di1_dt
        
        h1_0 = term1_h1 + term2_h1
        h2_0 = term1_h2 + term2_h2
        h3_0 = term1_h3 + term2_h3
        
        '''
        print(f"\n3. Momento angular h:")
        print(f"   h = [{h1_0:.6f}, {h2_0:.6f}, {h3_0:.6f}] rad/s")
        
        # Verificação
        h_dot_i = h1_0*i1_0 + h2_0*i2_0 + h3_0*i3_0
        expected = (I_P / I_T) * omega1_inertial
        print(f"   Verificação h·i' = {h_dot_i:.6f} vs esperado = {expected:.6f}")
        '''

        # Posição inicial (posição absoluta da arma)
        abs_position = self.weapon.get_absolute_position()
        x0, y0, z0 = abs_position
        
        # Monta vetor de estado
        y0_vec = np.array([V1_0, V2_0, V3_0,
                           h1_0, h2_0, h3_0,
                           i1_0, i2_0, i3_0,
                           x0, y0, z0], dtype=float)
        
        return y0_vec
    
    def rhs(self, t, y):
        """
        Lado direito das equações diferenciais (RHS).
        
        Parâmetros:
        -----------
        t : float
            Tempo
        y : array
            Vetor de estado [V1, V2, V3, h1, h2, h3, i1, i2, i3, x, y, z]
            
        Retorna:
        --------
        array : derivadas do vetor de estado
        """
        V1, V2, V3, h1, h2, h3, i1, i2, i3, x, ypos, z = y
        
        # Velocidade relativa (com vento)
        v1 = V1 - self.environment.W1
        v2 = V2 - self.environment.W2
        v3 = V3 - self.environment.W3
        v = sqrt(v1*v1 + v2*v2 + v3*v3)
        
        # Número de Mach
        mach = v / self.environment.sound_speed

        # Ângulo de ataque
        cos_alpha_t = (v1*i1 + v2*i2 + v3*i3) / v
        cos_alpha_t = np.clip(cos_alpha_t, -1.0, 1.0)
        alpha_rad = acos(cos_alpha_t)
        
        # Obter coeficientes aerodinâmicos
        coeffs = self.aero_coeffs.get_coefficients(mach, alpha_rad)
        
        C_D = coeffs['CD_total'] # Drag Force Coefficient
        C_Lalpha = coeffs['CLA_total'] # Lift Force Coefficient
        C_Npalpha = coeffs['CYP'] #Magnus Force Coefficient
        C_Nq = 0 # Pitching Dumping Force Coefficient
        C_Nalpha_dot = 0 # Pitching Dumping Force Coefficient (segunda componente)
        C_lp = coeffs['CLP'] # Spin Dumping Moment Coefficient
        C_Malpha = coeffs['CMA'] # Pitching Moment Coefficient
        C_Mpalpha = coeffs['CNP_total'] # Magnus Moment Coefficient
        C_Mq = coeffs['CMQ'] #Pitching Dumping Moment Coefficient
        C_Malpha_dot = 0
        
        C_l_delta = 0.0
        delta_F = 0.0
        
        # Parâmetros do projétil e ambiente
        m = self.projectile.mass
        S = self.projectile.S
        d = self.projectile.diameter
        I_P = self.projectile.I_P
        I_T = self.projectile.I_T
        rho = self.environment.rho
        g = self.environment.g
        
        # ω₁ = (I_T/I_P) (h · i')
        h_dot_i = (h1*i1 + h2*i2 + h3*i3)
        omega1 = (I_T/I_P) * h_dot_i
        
        # Equações de força (dV/dt)
        dV1 = (
            - (rho*v*S*C_D)/(2*m) * v1
            + (rho*S*C_Lalpha)/(2*m) * ( (v*v)*i1 - v*v1*cos_alpha_t )
            - (rho*S*d*C_Npalpha*omega1)/(2*m) * ( v3*i2 - v2*i3 )
            + (rho*v*S*d*(C_Nq + C_Nalpha_dot))/(2*m) * ( h2*i3 - h3*i2 )
        )
        dV2 = (
            - (rho*v*S*C_D)/(2*m) * v2
            + (rho*S*C_Lalpha)/(2*m) * ( (v*v)*i2 - v*v2*cos_alpha_t )
            - (rho*S*d*C_Npalpha*omega1)/(2*m) * ( v1*i3 - v3*i1 )
            + (rho*v*S*d*(C_Nq + C_Nalpha_dot))/(2*m) * ( h3*i1 - h1*i3 )
            - g
        )
        dV3 = (
            - (rho*v*S*C_D)/(2*m) * v3
            + (rho*S*C_Lalpha)/(2*m) * ( (v*v)*i3 - v*v3*cos_alpha_t )
            - (rho*S*d*C_Npalpha*omega1)/(2*m) * ( v2*i1 - v1*i2 )
            + (rho*v*S*d*(C_Nq + C_Nalpha_dot))/(2*m) * ( h1*i2 - h2*i1 )
        )
        
        # Equações de momento (dh/dt)
        dh1 = (
            (rho*v*S*d**2*C_lp*omega1)/(2*I_T) * i1
            + (rho*v**2*S*d*delta_F*C_l_delta)/(2*I_T) * i1 
            + (rho*v*S*d*C_Malpha)/(2*I_T) * ( v2*i3 - v3*i2 )
            + (rho*S*d**2*C_Mpalpha*omega1)/(2*I_T) * (v1 - v*i1*cos_alpha_t)
            + (rho*v*S*d**2*(C_Mq+C_Malpha_dot))/(2*I_T) * ( h1 - ((I_P/I_T)*omega1)*i1 )
        )
        dh2 = (
            (rho*v*S*d**2*C_lp*omega1)/(2*I_T) * i2
            + (rho*v**2*S*d*delta_F*C_l_delta)/(2*I_T) * i2
            + (rho*v*S*d*C_Malpha)/(2*I_T) * ( v3*i1 - v1*i3 )
            + (rho*S*d**2*C_Mpalpha*omega1)/(2*I_T) * (v2 - v*i2*cos_alpha_t)
            + (rho*v*S*d**2*(C_Mq+C_Malpha_dot))/(2*I_T) * ( h2 - ((I_P/I_T)*omega1)*i2 )
        )
        dh3 = (
            (rho*v*S*d**2*C_lp*omega1)/(2*I_T) * i3
            + (rho*v**2*S*d*delta_F*C_l_delta)/(2*I_T) * i3 
            + (rho*v*S*d*C_Malpha)/(2*I_T) * ( v1*i2 - v2*i1 )
            + (rho*S*d**2*C_Mpalpha*omega1)/(2*I_T) * (v3 - v*i3*cos_alpha_t)
            + (rho*v*S*d**2*(C_Mq+C_Malpha_dot))/(2*I_T) * ( h3 - ((I_P/I_T)*omega1)*i3 )
        )
        
        # Equação de orientação (di'/dt)
        di1 = h2*i3 - h3*i2
        di2 = h3*i1 - h1*i3
        di3 = h1*i2 - h2*i1
        
        # Equação de posição
        dx, dy, dz = V1, V2, V3
        
        return np.array([dV1, dV2, dV3, dh1, dh2, dh3, di1, di2, di3, dx, dy, dz], dtype=float)
    
    def simulate(self, max_time=100.0, alpha0_deg=0.0, beta0_deg=0.0,
                w_j0=5.0, w_k0=5.0, rtol=1e-7, atol=1e-8):
        """
        Executa a simulação balística.
        
        Parâmetros:
        -----------
        max_time : float
            Tempo máximo de simulação [s]
        alpha0_deg : float
            Ângulo de arfagem inicial [°]
        beta0_deg : float
            Ângulo de guinada inicial [°]
        w_j0 : float
            Velocidade angular em j' [rad/s]
        w_k0 : float
            Velocidade angular em k' [rad/s]
        rtol : float
            Tolerância relativa do integrador
        atol : float
            Tolerância absoluta do integrador
            
        Retorna:
        --------
        SimulationResult : objeto contendo os resultados
        """
        print("\n" + "="*80)
        print("INICIANDO SIMULAÇÃO")
        print("="*80)
        
        # Construir condições iniciais
        y0 = self.build_initial_conditions(alpha0_deg, beta0_deg, w_j0, w_k0)
        
        # Evento de impacto no solo
        def ground_event(t, y):
            return y[10]  # y position
        ground_event.direction = -1
        ground_event.terminal = True
        
        print("\nIntegrando trajetória...")
        
        # Resolver EDO
        sol = solve_ivp(self.rhs, (0.0, max_time), y0,
                       method='DOP853',
                       rtol=rtol, atol=atol,
                       events=ground_event,
                       max_step=0.1)
        
        if sol.success:
            print(f"✓ Integração bem-sucedida!")
            print(f"  Tempo de voo: {sol.t[-1]:.2f} s")
        else:
            print(f"✗ Erro na integração: {sol.message}")
        
        # Criar objeto de resultado
        self.result = SimulationResult(sol, self)
        
        return self.result


# =============================================================================
# CLASSE RESULTADO DA SIMULAÇÃO
# ================================================================  =============

class SimulationResult:
    """
    Classe que armazena e analisa os resultados de uma simulação balística.
    Gráficos no estilo 3Blue1Brown com fundo branco.
    """
    
    # Paleta de cores estilo 3Blue1Brown - FUNDO BRANCO
    COLORS = {
        'bg': '#FFFFFF',           # Fundo branco
        'primary': '#1A7FA0',      # Azul cyan mais escuro
        'secondary': '#2E5F7D',    # Azul esverdeado escuro
        'accent1': '#3BA3C7',      # Azul claro vibrante
        'accent2': '#1A5F7A',      # Azul petróleo
        'grid': '#D0D0D0',         # Grid visível mas suave
        'text': '#1A1A1A',         # Texto escuro
        'green': '#2E7D32',        # Verde escuro
        'red': '#C62828',          # Vermelho escuro
        'yellow': '#F9A825',       # Amarelo dourado
        'purple': '#6A1B9A',       # Roxo escuro
        'orange': '#EF6C00'        # Laranja escuro
    }
    
    def __init__(self, solution, simulator):
        """
        Inicializa o resultado da simulação.
        
        Parâmetros:
        -----------
        solution : OdeResult
            Resultado do scipy.integrate.solve_ivp
        simulator : BallisticSimulator
            O simulador que gerou estes resultados
        """
        self.solution = solution
        self.simulator = simulator
        
        # Extrair dados
        self.t = solution.t
        self.V1, self.V2, self.V3 = solution.y[0:3]
        self.h1, self.h2, self.h3 = solution.y[3:6]
        self.i1, self.i2, self.i3 = solution.y[6:9]
        self.x, self.y, self.z = solution.y[9:12]
        
        # Calcular grandezas derivadas
        self._calculate_derived_quantities()
    
    def _calculate_derived_quantities(self):
        """Calcula grandezas derivadas da trajetória."""
        # Velocidade total
        self.V_mag = np.sqrt(self.V1**2 + self.V2**2 + self.V3**2)
        
        # Mach
        self.mach = self.V_mag / self.simulator.environment.sound_speed
        
        # Momento angular total
        self.h_mag = np.sqrt(self.h1**2 + self.h2**2 + self.h3**2)
        
        # Spin rate
        self.spin_rate = []
        I_P = self.simulator.projectile.I_P
        I_T = self.simulator.projectile.I_T
        for idx in range(len(self.t)):
            h_dot_i = self.h1[idx]*self.i1[idx] + self.h2[idx]*self.i2[idx] + self.h3[idx]*self.i3[idx]
            omega1 = (I_T/I_P) * h_dot_i
            self.spin_rate.append(omega1)
        self.spin_rate = np.array(self.spin_rate)
        
        # Ângulo de ataque
        self.alpha_traj = []
        for idx in range(len(self.t)):
            v1 = self.V1[idx] - self.simulator.environment.W1
            v2 = self.V2[idx] - self.simulator.environment.W2
            v3 = self.V3[idx] - self.simulator.environment.W3
            v = sqrt(v1*v1 + v2*v2 + v3*v3) + 1e-12
            
            v_along = v1*self.i1[idx] + v2*self.i2[idx] + v3*self.i3[idx]
            v_perp1 = v1 - v_along*self.i1[idx]
            v_perp2 = v2 - v_along*self.i2[idx]
            v_perp3 = v3 - v_along*self.i3[idx]
            v_perp = sqrt(v_perp1**2 + v_perp2**2 + v_perp3**2)
            
            alpha = np.degrees(atan2(v_perp, v_along))
            self.alpha_traj.append(alpha)
        self.alpha_traj = np.array(self.alpha_traj)
        
        # Estatísticas
        self.alcance_max = float(self.x[-1])
        self.altura_max = float(np.max(self.y))
        self.desvio_lateral_max = float(np.max(np.abs(self.z)))
        self.tempo_voo = float(self.t[-1])
    
    def _setup_3b1b_style(self, ax, title=''):
        """Configura o estilo 3Blue1Brown para um eixo (fundo branco)."""
        ax.set_facecolor(self.COLORS['bg'])
        ax.grid(True, alpha=0.4, color=self.COLORS['grid'], linestyle='-', linewidth=0.8)
        ax.tick_params(colors=self.COLORS['text'], labelsize=10)
        ax.spines['bottom'].set_color(self.COLORS['text'])
        ax.spines['top'].set_color(self.COLORS['text']) 
        ax.spines['right'].set_color(self.COLORS['text'])
        ax.spines['left'].set_color(self.COLORS['text'])
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.xaxis.label.set_color(self.COLORS['text'])
        ax.yaxis.label.set_color(self.COLORS['text'])
        ax.title.set_color(self.COLORS['text'])
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    def print_statistics(self):
        """Imprime estatísticas da trajetória."""
        print(f"\n{'='*80}")
        print("ESTATÍSTICAS DA TRAJETÓRIA")
        print(f"{'='*80}")
        print(f"  Alcance: {self.alcance_max/1000:.2f} km")
        print(f"  Altura máxima: {self.altura_max/1000:.2f} km")
        print(f"  Desvio lateral: {self.desvio_lateral_max:.2f} m")
        print(f"  Tempo de voo: {self.tempo_voo:.2f} s")
        print(f"\nÂNGULO DE ATAQUE:")
        print(f"  Mínimo: {np.min(self.alpha_traj):.2f}°")
        print(f"  Máximo: {np.max(self.alpha_traj):.2f}°")
        print(f"  Médio: {np.mean(self.alpha_traj):.2f}°")
    
    def plot_trajectory_3d(self, save_path='01_trajectory_3d_white.png'):
        """Gráfico 1: Trajetória 3D completa."""
        fig = plt.figure(figsize=(12, 10), facecolor=self.COLORS['bg'])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(self.COLORS['bg'])
        
        # Trajetória
        ax.plot(self.x/1000, self.z, self.y/1000, 
                color=self.COLORS['primary'], linewidth=3, alpha=0.9)
        
        # Pontos de início e fim
        ax.scatter([0], [0], [0], c=self.COLORS['green'], 
                  s=200, marker='o', edgecolors=self.COLORS['text'], linewidths=2.5, label='Início')
        ax.scatter([self.x[-1]/1000], [self.z[-1]], [self.y[-1]/1000], 
                  c=self.COLORS['red'], s=200, marker='X', 
                  edgecolors=self.COLORS['text'], linewidths=2.5, label='Impacto')
        
        # Labels e título
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold', 
                     color=self.COLORS['text'], labelpad=10)
        ax.set_ylabel('Deriva Lateral [m]', fontsize=12, fontweight='bold',
                     color=self.COLORS['text'], labelpad=10)
        ax.set_zlabel('Altitude [km]', fontsize=12, fontweight='bold',
                     color=self.COLORS['text'], labelpad=10)
        ax.set_title('Trajetória 3D Completa', fontsize=16, fontweight='bold', 
                    color=self.COLORS['text'], pad=20)
        
        # Estilo dos eixos
        ax.tick_params(colors=self.COLORS['text'])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(self.COLORS['grid'])
        ax.yaxis.pane.set_edgecolor(self.COLORS['grid'])
        ax.zaxis.pane.set_edgecolor(self.COLORS['grid'])
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
        
        # Legenda
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          loc='upper left', fontsize=10, framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_top_view(self, save_path='02_top_view_white.png'):
        """Gráfico 2: Vista de cima (deriva lateral)."""
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Vista de Cima - Deriva Lateral')
        
        # Trajetória
        ax.plot(self.x/1000, self.z, color=self.COLORS['primary'], 
               linewidth=3, alpha=0.9)
        
        # Pontos
        ax.scatter([self.x[0]/1000], [self.z[0]], c=self.COLORS['green'], 
                  s=200, marker='o', edgecolors=self.COLORS['text'], 
                  linewidths=2.5, label='Início', zorder=5)
        ax.scatter([self.x[-1]/1000], [self.z[-1]], c=self.COLORS['red'], 
                  s=200, marker='X', edgecolors=self.COLORS['text'], 
                  linewidths=2.5, label='Impacto', zorder=5)
        
        # Linha de referência (sem deriva)
        ax.axhline(y=0, color=self.COLORS['yellow'], linestyle='--', 
                  alpha=0.7, linewidth=2, label='Referência (Z=0)')
        
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Deriva Lateral [m]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_side_view(self, save_path='03_side_view_white.png'):
        """Gráfico 3: Vista lateral (perfil vertical)."""
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Vista Lateral - Perfil Vertical')
        
        # Área sob a curva
        ax.fill_between(self.x/1000, 0, self.y/1000, 
                       alpha=0.15, color=self.COLORS['primary'])
        
        # Trajetória
        ax.plot(self.x/1000, self.y/1000, color=self.COLORS['primary'], 
               linewidth=3, alpha=0.9)
        
        # Pontos
        ax.scatter([self.x[0]/1000], [self.y[0]/1000], c=self.COLORS['green'], 
                  s=200, marker='o', edgecolors=self.COLORS['text'], 
                  linewidths=2.5, label='Início', zorder=5)
        ax.scatter([self.x[-1]/1000], [self.y[-1]/1000], c=self.COLORS['red'], 
                  s=200, marker='X', edgecolors=self.COLORS['text'], 
                  linewidths=2.5, label='Impacto', zorder=5)
        
        # Linha do solo
        ax.axhline(y=0, color=self.COLORS['text'], linestyle='-', 
                  alpha=0.8, linewidth=2.5, label='Nível do Solo')
        
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Altitude [km]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_velocity_vs_time(self, save_path='04_velocity_vs_time_white.png'):
        """Gráfico 4a: Velocidade vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Velocidade vs Tempo')
        
        ax.plot(self.t, self.V_mag, color=self.COLORS['primary'], 
               linewidth=3.5, label='|V| (Magnitude)', alpha=0.9)
        ax.plot(self.t, self.V1, color=self.COLORS['accent1'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='V₁ (Forward)')
        ax.plot(self.t, self.V2, color=self.COLORS['green'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='V₂ (Up)')
        ax.plot(self.t, self.V3, color=self.COLORS['purple'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='V₃ (Right)')
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Velocidade [m/s]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_velocity_vs_distance(self, save_path='05_velocity_vs_distance_white.png'):
        """Gráfico 4b: Velocidade vs Distância."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Velocidade vs Distância')
        
        ax.plot(self.x/1000, self.V_mag, color=self.COLORS['primary'], 
               linewidth=3.5, label='|V| (Magnitude)', alpha=0.9)
        ax.plot(self.x/1000, self.V1, color=self.COLORS['accent1'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='V₁ (Forward)')
        ax.plot(self.x/1000, self.V2, color=self.COLORS['green'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='V₂ (Up)')
        ax.plot(self.x/1000, self.V3, color=self.COLORS['purple'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='V₃ (Right)')
        
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Velocidade [m/s]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_axis_orientation_vs_time(self, save_path='06_axis_orientation_vs_time_white.png'):
        """Gráfico 5a: Orientação do eixo i' vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, "Evolução do Eixo Polar i' vs Tempo")
        
        ax.plot(self.t, self.i1, color=self.COLORS['red'], 
               linewidth=3, label="i'₁", alpha=0.9)
        ax.plot(self.t, self.i2, color=self.COLORS['green'], 
               linewidth=3, label="i'₂", alpha=0.9)
        ax.plot(self.t, self.i3, color=self.COLORS['accent1'], 
               linewidth=3, label="i'₃", alpha=0.9)
        
        ax.axhline(y=0, color=self.COLORS['text'], linestyle=':', 
                  linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel("Componentes de i'", fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=11, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_axis_orientation_vs_distance(self, save_path='07_axis_orientation_vs_distance_white.png'):
        """Gráfico 5b: Orientação do eixo i' vs Distância."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, "Evolução do Eixo Polar i' vs Distância")
        
        ax.plot(self.x/1000, self.i1, color=self.COLORS['red'], 
               linewidth=3, label="i'₁", alpha=0.9)
        ax.plot(self.x/1000, self.i2, color=self.COLORS['green'], 
               linewidth=3, label="i'₂", alpha=0.9)
        ax.plot(self.x/1000, self.i3, color=self.COLORS['accent1'], 
               linewidth=3, label="i'₃", alpha=0.9)
        
        ax.axhline(y=0, color=self.COLORS['text'], linestyle=':', 
                  linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel("Componentes de i'", fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=11, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_angular_momentum_vs_time(self, save_path='08_angular_momentum_vs_time_white.png'):
        """Gráfico 6a: Momento Angular vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Momento Angular vs Tempo')
        
        ax.plot(self.t, self.h_mag, color=self.COLORS['primary'], 
               linewidth=3.5, label='|h| (Magnitude)', alpha=0.9)
        ax.plot(self.t, self.h1, color=self.COLORS['red'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='h₁')
        ax.plot(self.t, self.h2, color=self.COLORS['green'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='h₂')
        ax.plot(self.t, self.h3, color=self.COLORS['accent1'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='h₃')
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Momento Angular [rad/s]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_angular_momentum_vs_distance(self, save_path='09_angular_momentum_vs_distance_white.png'):
        """Gráfico 6b: Momento Angular vs Distância."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Momento Angular vs Distância')
        
        ax.plot(self.x/1000, self.h_mag, color=self.COLORS['primary'], 
               linewidth=3.5, label='|h| (Magnitude)', alpha=0.9)
        ax.plot(self.x/1000, self.h1, color=self.COLORS['red'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='h₁')
        ax.plot(self.x/1000, self.h2, color=self.COLORS['green'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='h₂')
        ax.plot(self.x/1000, self.h3, color=self.COLORS['accent1'], 
               linewidth=2.5, linestyle='--', alpha=0.8, label='h₃')
        
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Momento Angular [rad/s]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_angle_of_attack_vs_time(self, save_path='10_angle_of_attack_vs_time_white.png'):
        """Gráfico 7a: Ângulo de Ataque vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Ângulo de Ataque vs Tempo')
        
        ax.plot(self.t, self.alpha_traj, color=self.COLORS['primary'], 
               linewidth=3, alpha=0.9)
        
        # Preencher áreas positivas e negativas
        ax.fill_between(self.t, 0, self.alpha_traj, 
                       where=(self.alpha_traj >= 0),
                       color=self.COLORS['green'], alpha=0.25, label='α positivo')
        ax.fill_between(self.t, 0, self.alpha_traj, 
                       where=(self.alpha_traj < 0),
                       color=self.COLORS['red'], alpha=0.25, label='α negativo')
        
        ax.axhline(y=0, color=self.COLORS['text'], linestyle=':', 
                  linewidth=1.5, alpha=0.6)
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ângulo de Ataque [°]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_angle_of_attack_vs_distance(self, save_path='11_angle_of_attack_vs_distance_white.png'):
        """Gráfico 7b: Ângulo de Ataque vs Distância."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Ângulo de Ataque vs Distância')
        
        ax.plot(self.x/1000, self.alpha_traj, color=self.COLORS['primary'], 
               linewidth=3, alpha=0.9)
        
        # Preencher áreas positivas e negativas
        ax.fill_between(self.x/1000, 0, self.alpha_traj, 
                       where=(self.alpha_traj >= 0),
                       color=self.COLORS['green'], alpha=0.25, label='α positivo')
        ax.fill_between(self.x/1000, 0, self.alpha_traj, 
                       where=(self.alpha_traj < 0),
                       color=self.COLORS['red'], alpha=0.25, label='α negativo')
        
        ax.axhline(y=0, color=self.COLORS['text'], linestyle=':', 
                  linewidth=1.5, alpha=0.6)
        
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ângulo de Ataque [°]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_mach_vs_time(self, save_path='12_mach_vs_time_white.png'):
        """Gráfico 8a: Número de Mach vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Número de Mach vs Tempo')
        
        ax.plot(self.t, self.mach, color=self.COLORS['purple'], 
               linewidth=3, alpha=0.9, label='Mach')
        
        ax.axhline(y=1.0, color=self.COLORS['red'], linestyle='--', 
                  alpha=0.8, linewidth=2.5, label='Mach 1 (Barreira do Som)')
        
        # Preencher regiões
        ax.fill_between(self.t, 0, self.mach, 
                       where=(self.mach >= 1.0),
                       color=self.COLORS['red'], alpha=0.15, label='Supersônico')
        ax.fill_between(self.t, 0, self.mach, 
                       where=(self.mach < 1.0),
                       color=self.COLORS['green'], alpha=0.15, label='Subsônico')
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Mach', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_mach_vs_distance(self, save_path='13_mach_vs_distance_white.png'):
        """Gráfico 8b: Número de Mach vs Distância."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Número de Mach vs Distância')
        
        ax.plot(self.x/1000, self.mach, color=self.COLORS['purple'], 
               linewidth=3, alpha=0.9, label='Mach')
        
        ax.axhline(y=1.0, color=self.COLORS['red'], linestyle='--', 
                  alpha=0.8, linewidth=2.5, label='Mach 1 (Barreira do Som)')
        
        # Preencher regiões
        ax.fill_between(self.x/1000, 0, self.mach, 
                       where=(self.mach >= 1.0),
                       color=self.COLORS['red'], alpha=0.15, label='Supersônico')
        ax.fill_between(self.x/1000, 0, self.mach, 
                       where=(self.mach < 1.0),
                       color=self.COLORS['green'], alpha=0.15, label='Subsônico')
        
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Mach', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_spin_rate_vs_time(self, save_path='14_spin_rate_vs_time_white.png'):
        """Gráfico 9a: Taxa de Spin vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Taxa de Spin vs Tempo')
        
        omega_spin_inicial = self.simulator.projectile.calculate_initial_spin(
            self.simulator.weapon.muzzle_velocity)
        
        ax.plot(self.t, self.spin_rate, color=self.COLORS['orange'], 
               linewidth=3, alpha=0.9, label='Spin Atual')
        
        ax.axhline(y=omega_spin_inicial, color=self.COLORS['yellow'], 
                  linestyle=':', alpha=0.8, linewidth=2.5, label='Spin Inicial')
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taxa de Spin [rad/s]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_spin_rate_vs_distance(self, save_path='15_spin_rate_vs_distance_white.png'):
        """Gráfico 9b: Taxa de Spin vs Distância."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Taxa de Spin vs Distância')
        
        omega_spin_inicial = self.simulator.projectile.calculate_initial_spin(
            self.simulator.weapon.muzzle_velocity)
        
        ax.plot(self.x/1000, self.spin_rate, color=self.COLORS['orange'], 
               linewidth=3, alpha=0.9, label='Spin Atual')
        
        ax.axhline(y=omega_spin_inicial, color=self.COLORS['yellow'], 
                  linestyle=':', alpha=0.8, linewidth=2.5, label='Spin Inicial')
        
        ax.set_xlabel('Alcance [km]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taxa de Spin [rad/s]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    
    def plot_altitude_vs_time(self, save_path='16_altitude_vs_time_white.png'):
        """Gráfico 10: Altura (Altitude) vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Altitude vs Tempo')
        
        # Área sob a curva
        ax.fill_between(self.t, 0, self.y/1000, 
                       alpha=0.15, color=self.COLORS['primary'])
        
        # Linha da altitude
        ax.plot(self.t, self.y/1000, color=self.COLORS['primary'], 
               linewidth=3, alpha=0.9, label='Altitude')
        
        # Ponto de altura máxima
        idx_max_alt = np.argmax(self.y)
        ax.scatter([self.t[idx_max_alt]], [self.y[idx_max_alt]/1000], 
                  c=self.COLORS['red'], s=200, marker='*', 
                  edgecolors=self.COLORS['text'], linewidths=2.5, 
                  label=f'Altura Máxima: {self.altura_max/1000:.2f} km', zorder=5)
        
        # Linha do solo
        ax.axhline(y=0, color=self.COLORS['text'], linestyle='-', 
                  alpha=0.8, linewidth=2.5, label='Nível do Solo')
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Altitude [km]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_lateral_drift_vs_time(self, save_path='17_lateral_drift_vs_time_white.png'):
        """Gráfico 11: Desvio Lateral vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Desvio Lateral vs Tempo')
        
        # Linha do desvio lateral
        ax.plot(self.t, self.z, color=self.COLORS['accent1'], 
               linewidth=3, alpha=0.9, label='Desvio Lateral (Z)')
        
        # Preencher área acima/abaixo de zero
        ax.fill_between(self.t, 0, self.z, 
                       where=(self.z >= 0),
                       color=self.COLORS['green'], alpha=0.2, label='Desvio positivo')
        ax.fill_between(self.t, 0, self.z, 
                       where=(self.z < 0),
                       color=self.COLORS['red'], alpha=0.2, label='Desvio negativo')
        
        # Linha de referência (sem deriva)
        ax.axhline(y=0, color=self.COLORS['yellow'], linestyle='--', 
                  alpha=0.7, linewidth=2, label='Referência (Z=0)')
        
        # Desvio máximo
        idx_max_z = np.argmax(np.abs(self.z))
        ax.scatter([self.t[idx_max_z]], [self.z[idx_max_z]], 
                  c=self.COLORS['purple'], s=200, marker='D', 
                  edgecolors=self.COLORS['text'], linewidths=2.5, 
                  label=f'Desvio Máx: {np.abs(self.z[idx_max_z]):.2f} m', zorder=5)
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Desvio Lateral [m]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_range_vs_time(self, save_path='18_range_vs_time_white.png'):
        """Gráfico 12: Alcance vs Tempo."""
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.COLORS['bg'])
        self._setup_3b1b_style(ax, 'Alcance vs Tempo')
        
        # Área sob a curva
        ax.fill_between(self.t, 0, self.x/1000, 
                       alpha=0.15, color=self.COLORS['secondary'])
        
        # Linha do alcance
        ax.plot(self.t, self.x/1000, color=self.COLORS['secondary'], 
               linewidth=3, alpha=0.9, label='Alcance (X)')
        
        # Ponto final (alcance máximo)
        ax.scatter([self.t[-1]], [self.x[-1]/1000], 
                  c=self.COLORS['red'], s=200, marker='X', 
                  edgecolors=self.COLORS['text'], linewidths=2.5, 
                  label=f'Alcance Final: {self.alcance_max/1000:.2f} km', zorder=5)
        
        ax.set_xlabel('Tempo [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Alcance [km]', fontsize=12, fontweight='bold')
        
        legend = ax.legend(facecolor=self.COLORS['bg'], edgecolor=self.COLORS['text'], 
                          fontsize=10, loc='best', framealpha=1)
        for text in legend.get_texts():
            text.set_color(self.COLORS['text'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor=self.COLORS['bg'], bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.show()
    
    def plot_all_graphs(self):
        """Gera todos os gráficos individuais."""
        print("\n" + "="*80)
        print("GERANDO TODOS OS GRÁFICOS (ESTILO 3BLUE1BROWN - FUNDO BRANCO)")
        print("="*80)
        
        self.plot_trajectory_3d()
        self.plot_top_view()
        self.plot_side_view()
        self.plot_velocity_vs_time()
        self.plot_velocity_vs_distance()
        self.plot_axis_orientation_vs_time()
        self.plot_axis_orientation_vs_distance()
        self.plot_angular_momentum_vs_time()
        self.plot_angular_momentum_vs_distance()
        self.plot_angle_of_attack_vs_time()
        self.plot_angle_of_attack_vs_distance()
        self.plot_mach_vs_time()
        self.plot_mach_vs_distance()
        self.plot_spin_rate_vs_time()
        self.plot_spin_rate_vs_distance()
        
        # NOVOS GRÁFICOS
        self.plot_altitude_vs_time()
        self.plot_lateral_drift_vs_time()
        self.plot_range_vs_time()
        
        print("\n✓ Todos os gráficos foram gerados!")

