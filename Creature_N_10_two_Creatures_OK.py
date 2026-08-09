import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import sys

try:
    import winsound
except ImportError:
    winsound = None

# ============================================================
# CONSTANTS
# ============================================================
N = 25
ANGULOS = np.linspace(0, 2 * np.pi, N, endpoint=False)
TIME_WARP = 1.0

# Softmax relaxation defaults
ALPHA_DEFAULT, T_DEFAULT = 0.35, 0.25

# N+3 hysteresis
PRED_PERCEPTION_RADIUS = 0.45
TAU_N3 = 16.0
K_ALERTA_SUBIDA = 0.9

# Maturation
EDAD_DESPIERTE_N2 = 0.6
EDAD_DESPIERTE_N4 = 1.0
EDAD_DESPIERTE_N5 = 1.5
RHO_MADURACION = 12.0

# Continuous loading
PICKUP_RADIUS = 0.12
NIDO_RADIUS = 0.25
K_PICKUP = 0.55
K_DROP = 0.45
STEEPNESS_GATE = 50.0
IMPULSO_MINIMO = 0.15

# World
MATERIALES_MAXIMOS = 9
MAX_MATERIALES_GENERADOS = 30

# N+4 learning
TAU_TRACE = 20.0
LR_N4 = 0.02
THRESHOLD_OUTCOME = 0.15

# N+5 learning
TAU_TRACE_N5 = 25.0
LR_N5 = 0.015
THRESHOLD_SOCIAL = 0.10

# ============================================================
# UTILITIES
# ============================================================
def _compuerta(x, filo=30.0):
    z = np.clip(-filo * x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))

def relajar_softmax(actividad, energias, alpha, temperatura):
    e = energias - np.max(energias)
    pesos = np.exp(e / temperatura)
    objetivo = pesos / np.sum(pesos)
    return (1.0 - alpha) * actividad + alpha * objetivo

# ============================================================
# WORLD STATE
# ============================================================
class WorldState:
    def __init__(self, start_pos=None):
        self.reset(start_pos)

    def reset(self, start_pos=None):
        self.pos = start_pos if start_pos is not None else np.array([0.0, 0.0])
        self.theta = 0.0
        self.hunger = 0.3
        self.safety = 0.0
        self.danger = 0.0
        self.border_stress = 0.0
        self.alerta_n3 = 0.0
        self.edad = 0.0
        self.L = 0.0
        self.total_depositado = 0.0
        self.total_recogido_frac = 0.0
        self.impulso_construir = 0.8
        self.urgencia_constructiva = 1.2
        self.instinto_construccion = 0.0
        self.instinto_aprendizaje = 0.0
        self.instinto_social = 0.0
        self.tiempo_sin_material = 0.0
        self.tiempo_sin_construir = 0.0
        self.nido_completado = False
        self.material_activo = False
        self.materiales_generados = 0
        self.food_lock = False
        self.home_lock = False

        self.food_pos = np.array([0.65, 0.45])
        self.home_pos = np.array([-0.55, -0.35])
        self.pred_pos = np.array([0.0, -0.7])
        self.material_pos = np.array([0.0, 0.0])

        self.food_theta = 0.0
        self.home_theta = np.pi
        self.pred_theta = 0.0
        self.food_radius = 0.65
        self.home_radius = 0.60
        self.pred_radius = 0.8

        self.prev_hunger = 0.3
        self.prev_safety = 0.0
        self.prev_danger = 0.0

        self.effective_hunger = 0.3
        self.current_safety_need = 0.0
        self.reposo_intensidad = 0.0

        # Peer state (set by Simulation before brain step)
        self.peer_pos = None
        self.peer_hunger = 0.0
        self.peer_alert = 0.0
        self.peer_L = 0.0
        self.peer_theta = 0.0
        self.peer_dist = 999.0
        self.peer_angle = 0.0

    def angle_to(self, entity):
        if entity == 'food':
            target = self.food_pos
        elif entity == 'home':
            target = self.home_pos
        elif entity == 'predator':
            target = self.pred_pos
        elif entity == 'material':
            target = self.material_pos
        elif entity == 'nido':
            target = np.array([-0.70, 0.70])
        elif entity == 'peer':
            target = self.peer_pos if self.peer_pos is not None else self.pos + np.array([1.0, 0.0])
        else:
            return 0.0
        vec = target - self.pos
        return np.arctan2(vec[1], vec[0])

    def dist_to(self, entity):
        if entity == 'food':
            return np.linalg.norm(self.pos - self.food_pos)
        elif entity == 'home':
            return np.linalg.norm(self.pos - self.home_pos)
        elif entity == 'predator':
            return np.linalg.norm(self.pos - self.pred_pos)
        elif entity == 'material':
            return np.linalg.norm(self.pos - self.material_pos) if self.material_activo else 999.0
        elif entity == 'nido':
            return np.linalg.norm(self.pos - np.array([-0.70, 0.70]))
        elif entity == 'peer':
            return self.peer_dist
        return 999.0

    def update_homeostasis(self, dt):
        dist_home = self.dist_to('home')
        compuerta_en_casa = _compuerta(0.25 - dist_home)
        compuerta_poco_hambre = _compuerta(0.40 - self.hunger)
        compuerta_seguridad_alta = _compuerta(self.safety - 0.80)
        compuerta_urgencia = _compuerta(self.hunger - 0.50) * _compuerta(0.40 - self.safety)

        atenuacion = np.clip((1.0 - self.safety) / 0.20, 0, 1)
        eh_atenuado = self.hunger * atenuacion
        sn_atenuado = self.safety * 1.8
        eh_urgente = self.hunger * 1.5
        sn_urgente = self.safety * 0.3
        eh_plano = self.hunger
        sn_plano = self.safety

        peso_atenuado = compuerta_seguridad_alta
        peso_urgente = compuerta_urgencia * (1.0 - peso_atenuado)
        peso_plano = np.clip(1.0 - peso_atenuado - peso_urgente, 0, 1)

        effective_hunger_lejos = peso_atenuado*eh_atenuado + peso_urgente*eh_urgente + peso_plano*eh_plano
        current_safety_need_lejos = peso_atenuado*sn_atenuado + peso_urgente*sn_urgente + peso_plano*sn_plano

        en_casa_activo = compuerta_en_casa * (1.0 - compuerta_poco_hambre)
        self.reposo_intensidad = compuerta_en_casa * compuerta_poco_hambre
        lejos = 1.0 - compuerta_en_casa

        self.effective_hunger = en_casa_activo*self.hunger + lejos*effective_hunger_lejos
        self.current_safety_need = self.reposo_intensidad*1.0 + lejos*current_safety_need_lejos

    def step(self, dt):
        # Homeostasis
        self.hunger = np.clip(self.hunger + 0.0015*dt, 0, 1)
        dist_home = self.dist_to('home')
        if dist_home < 0.25:
            self.safety = 0.0
        else:
            self.safety = np.clip(self.safety + 0.0020*dt, 0, 1)
        self.danger = np.clip(self.danger + 0.003*dt, 0, 1)

        self.update_homeostasis(dt)

        # Age
        self.edad += 0.01 * dt
        self.instinto_construccion = _compuerta(self.edad - EDAD_DESPIERTE_N2, filo=RHO_MADURACION)
        self.instinto_aprendizaje = _compuerta(self.edad - EDAD_DESPIERTE_N4, filo=RHO_MADURACION)
        self.instinto_social = _compuerta(self.edad - EDAD_DESPIERTE_N5, filo=RHO_MADURACION)

        # Border stress
        dist_to_wall_x = 1.0 - abs(self.pos[0])
        dist_to_wall_y = 1.0 - abs(self.pos[1])
        closest_wall_dist = min(dist_to_wall_x, dist_to_wall_y)
        self.border_stress = (np.clip((0.25 - closest_wall_dist)/0.25, 0, 1) ** 2) if closest_wall_dist < 0.25 else 0.0

        # Move entities
        self.food_theta += 0.015*dt
        self.food_pos = np.array([self.food_radius*np.cos(self.food_theta), 0.55*np.sin(1.7*self.food_theta)])
        self.home_theta -= 0.010*dt
        self.home_pos = np.array([self.home_radius*np.cos(self.home_theta), 0.45*np.sin(1.3*self.home_theta)])

        # Predator position is updated by Simulation, not here

        # N+3 hysteresis
        dist_pred = self.dist_to('predator')
        percepcion_amenaza = _compuerta(PRED_PERCEPTION_RADIUS - dist_pred, filo=25.0)
        subida_alerta = K_ALERTA_SUBIDA * percepcion_amenaza * (1.0 - self.alerta_n3)
        bajada_alerta = (self.alerta_n3 / TAU_N3) * (1.0 - percepcion_amenaza)
        self.alerta_n3 = np.clip(self.alerta_n3 + dt*(subida_alerta - bajada_alerta), 0, 1)

# ============================================================
# TISSUE BASE CLASS
# ============================================================
class Tissue:
    def __init__(self, name, alpha, temperature, awakening_age=0.0, color='white'):
        self.name = name
        self.alpha = alpha
        self.temperature = temperature
        self.awakening_age = awakening_age
        self.color = color
        self.awake = False
        self.age_triggered = False
        self.activity = np.ones(N) / N
        self.energies = np.zeros(N)
        self.cms = self._init_cms()

    def _init_cms(self):
        return [{"angle": a, "explore": np.random.uniform(0, 0.05)} for a in ANGULOS]

    def try_awaken(self, edad):
        if not self.age_triggered and edad >= self.awakening_age:
            self.awake = True
            self.age_triggered = True
            print(f"[AWAKENING] {self.name} ignites at age {edad:.2f}")

    def compute_energies(self, world, modulatory):
        raise NotImplementedError

    def update(self, dt, world, modulatory):
        if not self.awake:
            return self.activity
        self.energies = self.compute_energies(world, modulatory)
        self.activity = relajar_softmax(self.activity, self.energies, self.alpha, self.temperature)
        return self.activity

    def get_motor_vector(self):
        vecs = np.stack([np.cos(ANGULOS), np.sin(ANGULOS)], axis=1)
        motor = self.activity @ vecs
        norm = np.linalg.norm(motor)
        return motor / norm if norm > 0 else np.zeros(2)

    def get_parameter_offsets(self):
        return {}

    def get_brain_gain(self, world):
        return 1.0

# ============================================================
# CONCRETE TISSUES
# ============================================================

class MOTTissue(Tissue):
    def __init__(self):
        super().__init__("MOT", 0.35, 0.25, awakening_age=0.0, color='#8B0000')
        for cm in self.cms:
            cm.update({
                "food_weight": np.random.uniform(1.2, 1.6),
                "home_weight": np.random.uniform(1.3, 1.7),
                "pred_weight": np.random.uniform(1.2, 1.7),
            })

    def compute_energies(self, world, modulatory):
        angle_food = world.angle_to('food')
        angle_home = world.angle_to('home')
        angle_pred = world.angle_to('predator')

        fw = np.array([cm["food_weight"] for cm in self.cms])
        hw = np.array([cm["home_weight"] for cm in self.cms])
        pw = np.array([cm["pred_weight"] for cm in self.cms])

        if 'n4_dw_food' in modulatory:
            fw += modulatory['n4_dw_food']
        if 'n4_dw_pred' in modulatory:
            pw += modulatory['n4_dw_pred']
        if 'n4_dw_home' in modulatory:
            hw += modulatory['n4_dw_home']

        return (
            world.effective_hunger * fw * np.cos(angle_food - ANGULOS)
            + world.current_safety_need * hw * np.cos(angle_home - ANGULOS)
            - world.alerta_n3 * pw * np.cos(angle_pred - ANGULOS)
            + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-1, 1, N)
            + 0.06 * np.random.randn(N)
        )

class NAVTissue(Tissue):
    def __init__(self):
        super().__init__("NAV", 0.40, 0.15, awakening_age=0.0, color='#00008B')
        for cm in self.cms:
            cm.update({"border_weight": np.random.uniform(0.8, 1.2)})

    def compute_energies(self, world, modulatory):
        fx = 1.0 / (1.0 + (world.pos[0] - (-0.95)))
        fe = 1.0 / (1.0 + (0.95 - world.pos[0]))
        fs = 1.0 / (1.0 + (world.pos[1] - (-0.95)))
        fn = 1.0 / (1.0 + (0.95 - world.pos[1]))
        vec = np.array([fe - fx, fn - fs])
        angle_border = np.arctan2(vec[1], vec[0])
        bw = np.array([cm["border_weight"] for cm in self.cms])
        ruido_reposo = np.random.uniform(-1, 1, N) * 2.0
        energies = -world.border_stress * bw * np.cos(angle_border - ANGULOS) + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-1, 1, N)
        return (1.0 - world.reposo_intensidad) * energies + world.reposo_intensidad * ruido_reposo

class FOOTissue(Tissue):
    def __init__(self):
        super().__init__("FOO", 0.35, 0.20, awakening_age=0.0, color='#006400')
        for cm in self.cms:
            cm.update({"food_gain": np.random.uniform(1.0, 1.5)})

    def compute_energies(self, world, modulatory):
        angle = world.angle_to('food')
        gw = np.array([cm["food_gain"] for cm in self.cms])
        return gw * np.cos(angle - ANGULOS) + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-0.5, 0.5, N)

class HOMTissue(Tissue):
    def __init__(self):
        super().__init__("HOM", 0.35, 0.20, awakening_age=0.0, color='#008B8B')
        for cm in self.cms:
            cm.update({"home_gain": np.random.uniform(1.0, 1.5)})

    def compute_energies(self, world, modulatory):
        angle = world.angle_to('home')
        gw = np.array([cm["home_gain"] for cm in self.cms])
        return gw * np.cos(angle - ANGULOS) + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-0.5, 0.5, N)

class THRTissue(Tissue):
    def __init__(self):
        super().__init__("THR", 0.35, 0.20, awakening_age=0.0, color='#FF8C00')
        for cm in self.cms:
            cm.update({"threat_gain": np.random.uniform(1.2, 1.8)})

    def compute_energies(self, world, modulatory):
        angle = world.angle_to('predator') + np.pi
        gw = np.array([cm["threat_gain"] for cm in self.cms])
        gain = np.clip((0.6 - world.dist_to('predator')) / 0.6, 0, 1) + world.alerta_n3 * 0.3
        return gain * gw * np.cos(angle - ANGULOS) + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-0.5, 0.5, N)

class EXPTissue(Tissue):
    def __init__(self):
        super().__init__("EXP", 0.30, 0.35, awakening_age=0.0, color='#4B0082')
        for cm in self.cms:
            cm.update({"novelty_gain": np.random.uniform(0.5, 1.0)})
        self.prev_pos = np.array([0.0, 0.0])

    def compute_energies(self, world, modulatory):
        gain = (1.0 - world.hunger) * (1.0 - world.alerta_n3) * 0.5
        ng = np.array([cm["novelty_gain"] for cm in self.cms])
        movement = world.pos - self.prev_pos
        angle_move = np.arctan2(movement[1], movement[0]) if np.linalg.norm(movement) > 0.001 else 0
        self.prev_pos = world.pos.copy()
        return gain * (ng * np.random.uniform(0, 1, N) + 0.5 * np.cos(angle_move - ANGULOS)) + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-1, 1, N)

class N2Tissue(Tissue):
    def __init__(self):
        super().__init__("N+2", 0.35, 0.20, awakening_age=EDAD_DESPIERTE_N2, color='#556B2F')
        for cm in self.cms:
            cm.update({
                "material_weight": np.random.uniform(1.2, 1.6),
                "nido_weight": np.random.uniform(1.3, 1.7),
            })

    def compute_energies(self, world, modulatory):
        # Material energy field: decays with distance, ~0 when no material (dist=999)
        material_field = np.exp(-world.dist_to('material') * 0.05)
        vacio_field = 1.0 - material_field

        angle_material = world.angle_to('material')
        angle_nido = world.angle_to('nido')
        mw = np.array([cm["material_weight"] for cm in self.cms])
        nw = np.array([cm["nido_weight"] for cm in self.cms])
        mat_align = np.cos(angle_material - ANGULOS)
        nid_align = np.cos(angle_nido - ANGULOS)

        return (
            (1 - world.L) * material_field * mw * mat_align * (0.8 + 0.5*world.instinto_construccion)
            + world.L * nw * nid_align * (0.7 + 0.5*world.instinto_construccion)
            + world.L * material_field * mw * mat_align * 0.1
            + (1 - world.L) * material_field * nw * nid_align * 0.1
            + (1 - world.L) * vacio_field * np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(0.5, 1.5, N) * (0.3 + world.instinto_construccion)
            + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-1, 1, N)
        )

class N3Tissue(Tissue):
    def __init__(self):
        super().__init__("N+3", 0.35, 0.20, awakening_age=0.0, color='#8B008B')
        for cm in self.cms:
            cm.update({"escape_weight": np.random.uniform(1.2, 1.7)})

    def compute_energies(self, world, modulatory):
        angle_evasion = world.angle_to('predator') + np.pi
        ew = np.array([cm["escape_weight"] for cm in self.cms])
        return (
            world.alerta_n3 * ew * np.cos(angle_evasion - ANGULOS)
            + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-1, 1, N)
            + 0.06 * np.random.randn(N)
        )

    def get_brain_gain(self, world):
        return 0.3 + 0.7 * world.alerta_n3

class N4Tissue(Tissue):
    def __init__(self):
        super().__init__("N+4", 0.30, 0.20, awakening_age=EDAD_DESPIERTE_N4, color='#696969')
        for cm in self.cms:
            cm.update({
                "dw_food": 0.0,
                "dw_pred": 0.0,
                "dw_home": 0.0,
                "trace": 0.0,
            })
        self.tau_trace = TAU_TRACE
        self.lr = LR_N4

    def compute_energies(self, world, modulatory):
        heading = world.theta
        return np.cos(heading - ANGULOS) + 0.1 * np.random.randn(N)

    def update(self, dt, world, modulatory):
        activity = super().update(dt, world, modulatory)
        if not self.awake:
            return activity

        outcome = (
            2.0 * (world.prev_hunger - world.hunger)
            + 1.5 * (world.safety - world.prev_safety)
            - 3.0 * (world.danger - world.prev_danger)
        )

        if abs(outcome) > THRESHOLD_OUTCOME:
            for k in range(N):
                self.cms[k]["trace"] *= (1.0 - 1.0 / self.tau_trace)
                self.cms[k]["trace"] += activity[k] * (1.0 - self.cms[k]["trace"])
                e = self.cms[k]["trace"]
                if e > 0.01:
                    self.cms[k]["dw_food"] = np.clip(
                        self.cms[k]["dw_food"] + self.lr * outcome * e * np.cos(world.angle_to('food') - ANGULOS[k]),
                        -0.6, 0.6
                    )
                    self.cms[k]["dw_pred"] = np.clip(
                        self.cms[k]["dw_pred"] + self.lr * outcome * e * (-np.cos(world.angle_to('predator') - ANGULOS[k])),
                        -0.6, 0.6
                    )
                    self.cms[k]["dw_home"] = np.clip(
                        self.cms[k]["dw_home"] + self.lr * outcome * e * np.cos(world.angle_to('home') - ANGULOS[k]),
                        -0.6, 0.6
                    )
        return activity

    def get_parameter_offsets(self):
        return {
            'n4_dw_food': np.array([cm["dw_food"] for cm in self.cms]),
            'n4_dw_pred': np.array([cm["dw_pred"] for cm in self.cms]),
            'n4_dw_home': np.array([cm["dw_home"] for cm in self.cms]),
        }

class INTTissue(Tissue):
    def __init__(self):
        super().__init__("INT", 0.30, 0.30, awakening_age=0.0, color='#C71585')
        for cm in self.cms:
            cm.update({"state_gain": np.random.uniform(0.8, 1.2)})

    def compute_energies(self, world, modulatory):
        angle_food = world.angle_to('food')
        angle_home = world.angle_to('home')
        angle_pred = world.angle_to('predator')
        sg = np.array([cm["state_gain"] for cm in self.cms])
        return (
            world.hunger * sg * np.cos(angle_food - ANGULOS)
            + world.safety * sg * np.cos(angle_home - ANGULOS)
            - world.alerta_n3 * sg * np.cos(angle_pred - ANGULOS)
            + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-0.3, 0.3, N)
        )

# ============================================================
# N+5: SOCIAL TISSUE
# ============================================================
class N5Tissue(Tissue):
    def __init__(self):
        super().__init__("N+5", 0.30, 0.20, awakening_age=EDAD_DESPIERTE_N5, color='#00BFFF')
        for cm in self.cms:
            cm.update({
                "w_empathy": 0.0,
                "w_mimic": 0.0,
                "w_avoid": 0.0,
                "w_cooperate": 0.0,
                "trace": 0.0,
            })
        self.tau_trace = TAU_TRACE_N5
        self.lr = LR_N5
        self.prev_social_quality = 0.0

    def compute_energies(self, world, modulatory):
        # Peer presence field: decays with distance, ~0 when no peer (dist=999)
        peer_presence = np.exp(-world.peer_dist * 0.05)

        angle_peer = world.peer_angle
        dist_peer = world.peer_dist

        # Peer state inference as continuous fields (not booleans)
        peer_hungry_field = _compuerta(world.peer_hunger - 0.5, filo=10.0)
        peer_alert_field = _compuerta(world.peer_alert - 0.3, filo=10.0)
        peer_constructing_field = _compuerta(world.peer_L - 0.3, filo=10.0)

        # Social drive: continuous, proportional to proximity
        proximity = np.clip(1.0 - dist_peer / 0.6, 0, 1)
        social_drive = proximity * (1.0 - world.alerta_n3 * 0.5)

        ew = np.array([cm["w_empathy"] for cm in self.cms])
        mw = np.array([cm["w_mimic"] for cm in self.cms])
        aw = np.array([cm["w_avoid"] for cm in self.cms])
        cw = np.array([cm["w_cooperate"] for cm in self.cms])

        # Base: approach peer (empathy + cooperate) vs avoid (avoid)
        approach = (ew + cw + mw) * social_drive
        avoidance = aw * (world.alerta_n3 + peer_alert_field * 0.5)

        # Everything scaled by peer_presence: if no peer, energy ~0
        return peer_presence * (
            approach * np.cos(angle_peer - ANGULOS)
            - avoidance * np.cos(angle_peer - ANGULOS)
            + np.array([cm["explore"] for cm in self.cms]) * np.random.uniform(-0.5, 0.5, N)
        )

    def update(self, dt, world, modulatory):
        activity = super().update(dt, world, modulatory)
        if not self.awake:
            return activity

        # Social outcome: did proximity to peer improve homeostasis?
        # When no peer, peer_dist=999 → proximity=0 → social_quality=0 → no learning
        dist_peer = world.peer_dist
        proximity = np.clip(1.0 - dist_peer / 0.6, 0, 1)
        homeostatic_delta = (
            (world.prev_hunger - world.hunger)
            + (world.safety - world.prev_safety)
            - (world.danger - world.prev_danger)
        )
        social_quality = proximity * homeostatic_delta

        if abs(social_quality) > THRESHOLD_SOCIAL:
            for k in range(N):
                self.cms[k]["trace"] *= (1.0 - 1.0 / self.tau_trace)
                self.cms[k]["trace"] += activity[k] * (1.0 - self.cms[k]["trace"])
                e = self.cms[k]["trace"]
                if e > 0.01:
                    angle_peer = world.peer_angle
                    align = np.cos(angle_peer - ANGULOS[k])
                    if social_quality > 0:
                        # Positive interaction: increase empathy and cooperation
                        self.cms[k]["w_empathy"] = np.clip(
                            self.cms[k]["w_empathy"] + self.lr * social_quality * e * align, -0.5, 0.5)
                        self.cms[k]["w_cooperate"] = np.clip(
                            self.cms[k]["w_cooperate"] + self.lr * social_quality * e * align, -0.5, 0.5)
                        self.cms[k]["w_mimic"] = np.clip(
                            self.cms[k]["w_mimic"] + self.lr * social_quality * e * align * 0.5, -0.5, 0.5)
                    else:
                        # Negative interaction: increase avoidance
                        self.cms[k]["w_avoid"] = np.clip(
                            self.cms[k]["w_avoid"] + self.lr * abs(social_quality) * e * align, -0.5, 0.5)

        self.prev_social_quality = social_quality
        return activity

    def get_parameter_offsets(self):
        return {
            'n5_w_empathy': np.array([cm["w_empathy"] for cm in self.cms]),
            'n5_w_avoid': np.array([cm["w_avoid"] for cm in self.cms]),
        }

# ============================================================
# BRAIN ORCHESTRATOR
# ============================================================
class Brain:
    REGIONS = {
        'mot': {'r0': 4, 'c0': 2,  'h': 4, 'w': 4, 'color': '#8B0000'},
        'nav': {'r0': 10, 'c0': 2,  'h': 4, 'w': 4, 'color': '#00008B'},
        'foo': {'r0': 16, 'c0': 2, 'h': 4, 'w': 4, 'color': '#006400'},
        'hom': {'r0': 22, 'c0': 2, 'h': 4, 'w': 4, 'color': '#008B8B'},
        'thr': {'r0': 4, 'c0': 8,  'h': 4, 'w': 4, 'color': '#FF8C00'},
        'exp': {'r0': 10, 'c0': 8,  'h': 4, 'w': 4, 'color': '#4B0082'},
        'n2':  {'r0': 16, 'c0': 8, 'h': 4, 'w': 4, 'color': '#556B2F'},
        'n3':  {'r0': 22, 'c0': 8, 'h': 4, 'w': 4, 'color': '#8B008B'},
        'n4':  {'r0': 4, 'c0': 14, 'h': 4, 'w': 4, 'color': '#696969'},
        'int': {'r0': 10, 'c0': 14, 'h': 4, 'w': 4, 'color': '#C71585'},
        'n5':  {'r0': 16, 'c0': 14, 'h': 4, 'w': 4, 'color': '#00BFFF'},
    }

    def __init__(self):
        self.tissues = {
            'mot': MOTTissue(),
            'nav': NAVTissue(),
            'foo': FOOTissue(),
            'hom': HOMTissue(),
            'thr': THRTissue(),
            'exp': EXPTissue(),
            'n2': N2Tissue(),
            'n3': N3Tissue(),
            'n4': N4Tissue(),
            'int': INTTissue(),
            'n5': N5Tissue(),
        }
        self.brain_map = np.zeros((26, 26))

    def step(self, dt, world):
        for tissue in self.tissues.values():
            tissue.try_awaken(world.edad)

        modulatory = {}
        if self.tissues['n4'].awake:
            modulatory.update(self.tissues['n4'].get_parameter_offsets())
        if self.tissues['n5'].awake:
            modulatory.update(self.tissues['n5'].get_parameter_offsets())

        for tissue in self.tissues.values():
            tissue.update(dt, world, modulatory)

        motor = self._integrate_motor(world)
        self._update_brain_map(world)
        return motor

    def _integrate_motor(self, world):
        m_mot = self.tissues['mot'].get_motor_vector()
        m_nav = self.tissues['nav'].get_motor_vector()
        m_foo = self.tissues['foo'].get_motor_vector()
        m_hom = self.tissues['hom'].get_motor_vector()
        m_thr = self.tissues['thr'].get_motor_vector()
        m_exp = self.tissues['exp'].get_motor_vector()
        m_n2 = self.tissues['n2'].get_motor_vector()
        m_n3 = self.tissues['n3'].get_motor_vector()
        m_n5 = self.tissues['n5'].get_motor_vector()

        w_mot = 1.0
        w_nav = world.border_stress
        w_foo = world.hunger * 0.5
        w_hom = world.current_safety_need * 0.5
        w_thr = np.clip((0.5 - world.dist_to('predator')) / 0.5, 0, 1) + world.alerta_n3 * 0.3
        w_exp = (1.0 - world.hunger) * (1.0 - world.alerta_n3) * 0.3

        g_impulso = _compuerta(world.impulso_construir - IMPULSO_MINIMO, filo=40.0)
        w_n2 = world.instinto_construccion * g_impulso * world.impulso_construir * 0.7 * (1.0 - world.hunger * 0.5)
        factor_peligro = np.clip(world.dist_to('predator') / 0.4, 0, 1)
        w_n2 *= factor_peligro
        w_n2 *= (1.0 + 0.3 * world.L)
        w_n2 = np.clip(w_n2, 0, 0.7)

        w_n3 = np.clip(world.alerta_n3 * 0.65, 0, 0.65)

        # N+5 social weight
        w_n5 = 0.0
        if world.peer_pos is not None:
            dist_peer = world.dist_to('peer')
            proximity = np.clip(1.0 - dist_peer / 0.6, 0, 1)
            w_n5 = world.instinto_social * proximity * 0.4
            w_n5 = np.clip(w_n5, 0, 0.4)

        suma = w_n2 + w_n3 + w_n5
        if suma > 1.0:
            factor = 1.0 / suma
            w_n2 *= factor
            w_n3 *= factor
            w_n5 *= factor

        motor = (
            (1.0 - w_n2 - w_n3 - w_n5) * (
                w_mot * m_mot +
                w_nav * m_nav +
                w_foo * m_foo +
                w_hom * m_hom +
                w_thr * m_thr +
                w_exp * m_exp
            ) / (w_mot + w_nav + w_foo + w_hom + w_thr + w_exp + 1e-8)
            + w_n2 * m_n2
            + w_n3 * m_n3
            + w_n5 * m_n5
        )

        if world.border_stress > 0:
            motor = (1.0 - world.border_stress) * motor + world.border_stress * m_nav

        norm = np.linalg.norm(motor)
        return motor / norm if norm > 0 else motor

    def _update_brain_map(self, world):
        reposo = _compuerta(0.25 - world.dist_to('home'), filo=30.0) * _compuerta(0.40 - world.hunger)
        decay_activo = 0.84
        decay_reposo = 0.70
        self.brain_map *= (decay_activo - (decay_activo - decay_reposo) * reposo)

        gain = 0.50 * (1.0 - 0.45 * reposo)
        noise_amp = 0.03

        for name, tissue in self.tissues.items():
            reg = self.REGIONS[name]
            r0, c0, h, w = reg['r0'], reg['c0'], reg['h'], reg['w']
            n_vis = h * w
            if len(tissue.activity) >= n_vis:
                vis = tissue.activity[:n_vis].reshape(h, w)
            else:
                vis = np.zeros((h, w))
            bg = tissue.get_brain_gain(world)
            self.brain_map[r0:r0+h, c0:c0+w] += vis * gain * bg + np.random.rand(h, w) * noise_amp

        bias = 0.03 * reposo
        self.brain_map += bias
        self.brain_map = np.clip(self.brain_map, 0, 0.80)

# ============================================================
# CREATURE
# ============================================================
class Creature:
    def __init__(self, name, color, start_pos, is_main=False):
        self.name = name
        self.color = color
        self.is_main = is_main
        self.world = WorldState(start_pos)
        self.brain = Brain()
        self.artists = {}
        self.alive = True

    def reset(self, start_pos):
        self.world.reset(start_pos)
        self.brain = Brain()
        self.alive = True
        try:
            if winsound:
                winsound.Beep(180, 600)
        except Exception:
            pass

    def update_world(self, dt, shared_world):
        w = self.world
        if not self.alive:
            return

        # Store history (saved BEFORE w.step modifies values)
        w.prev_hunger = w.hunger
        w.prev_safety = w.safety
        w.prev_danger = w.danger

        # Copy shared entity positions from shared_world (creature 1's world is authoritative)
        w.food_pos = shared_world.food_pos
        w.home_pos = shared_world.home_pos
        w.pred_pos = shared_world.pred_pos
        w.food_theta = shared_world.food_theta
        w.home_theta = shared_world.home_theta
        w.pred_theta = shared_world.pred_theta

        # Step world physics
        w.step(dt)

        dist_food = w.dist_to('food')
        dist_home = w.dist_to('home')
        dist_pred = w.dist_to('predator')
        dist_nido = w.dist_to('nido')
        dist_material = w.dist_to('material')

        # Collisions
        if dist_food < 0.10 and not w.food_lock:
            w.hunger = 0.0
            try:
                if winsound:
                    winsound.Beep(1200, 120)
            except Exception:
                pass
            w.food_lock = True
        if dist_home < 0.10 and not w.home_lock:
            try:
                if winsound:
                    winsound.Beep(500, 180)
            except Exception:
                pass
            w.home_lock = True
        if dist_pred < 0.12:
            if dist_home < 0.30:
                w.danger = 0.0
            else:
                self.alive = False
                return
        if dist_food > 0.18:
            w.food_lock = False
        if dist_home > 0.18:
            w.home_lock = False

        # Construction bookkeeping
        if not w.nido_completado:
            w.urgencia_constructiva += 0.005 * dt
            w.urgencia_constructiva += 0.015 * dt * _compuerta(dist_material - 999 + 1, filo=50.0) * (1.0 - w.L)
            w.urgencia_constructiva += 0.02 * dt * w.L
            progreso = w.total_depositado / MATERIALES_MAXIMOS
            w.urgencia_constructiva += 0.005 * progreso * dt
            w.tiempo_sin_construir += dt
            w.urgencia_constructiva += 0.01 * dt * _compuerta(w.tiempo_sin_construir - 60, filo=0.3)
            w.urgencia_constructiva = np.clip(w.urgencia_constructiva, 0, 1.5)

        if w.nido_completado:
            w.impulso_construir *= 0.95
        else:
            w.impulso_construir += 0.015 * dt
            w.impulso_construir += w.urgencia_constructiva * 0.02 * dt
            w.impulso_construir += 0.01 * dt * (1.0 - w.L) * (1.0 if w.material_activo else 0.0)
            w.impulso_construir += 0.02 * dt * w.L
            progreso = w.total_depositado / MATERIALES_MAXIMOS
            w.impulso_construir += 0.005 * progreso * dt
            w.impulso_construir += 0.01 * dt * _compuerta(w.tiempo_sin_construir - 50, filo=0.3)
        w.impulso_construir = np.clip(w.impulso_construir, 0, 1)

        # Continuous loading
        g_pickup = _compuerta(PICKUP_RADIUS - dist_material, filo=STEEPNESS_GATE) if w.material_activo else 0.0
        g_drop = _compuerta(NIDO_RADIUS - dist_nido, filo=STEEPNESS_GATE) if not w.nido_completado else 0.0

        flujo_entrada = K_PICKUP * g_pickup * (1.0 - w.L)
        flujo_salida = K_DROP * g_drop * w.L
        w.L = np.clip(w.L + dt * (flujo_entrada - flujo_salida), 0, 1)

        w.total_depositado = min(MATERIALES_MAXIMOS, w.total_depositado + dt * flujo_salida)
        w.total_recogido_frac += dt * flujo_entrada
        if w.total_recogido_frac >= 1.0 and w.material_activo:
            w.material_activo = False
            w.total_recogido_frac -= 1.0
            try:
                if winsound:
                    winsound.Beep(800, 100)
            except Exception:
                pass

        if w.total_depositado >= MATERIALES_MAXIMOS:
            w.nido_completado = True
            w.material_activo = False

        # Generate new material
        if not w.nido_completado and not w.material_activo:
            if w.materiales_generados < MAX_MATERIALES_GENERADOS:
                self._generar_material()
            else:
                w.tiempo_sin_material += dt
                if w.tiempo_sin_material > 150:
                    w.materiales_generados = 0
                    w.tiempo_sin_material = 0.0
                    self._generar_material()


    def _generar_material(self):
        w = self.world
        if w.materiales_generados >= MAX_MATERIALES_GENERADOS or w.nido_completado:
            w.material_activo = False
            return
        if w.material_activo:
            return
        centro_x, centro_y, radio = 0.75, -0.75, 0.20
        for _ in range(200):
            angulo = np.random.uniform(0, 2*np.pi)
            distancia = np.random.uniform(0, radio)
            nueva_pos = np.array([centro_x + distancia*np.cos(angulo), centro_y + distancia*np.sin(angulo)])
            if np.linalg.norm(nueva_pos - np.array([-0.70, 0.70])) > 0.20 and np.linalg.norm(nueva_pos - w.home_pos) > 0.20:
                w.material_pos = nueva_pos
                w.material_activo = True
                w.materiales_generados += 1
                return
        w.material_activo = False

    def apply_motor(self, motor, dt):
        w = self.world
        if not self.alive:
            return
        mag = np.linalg.norm(motor)
        if mag > 0:
            w.theta = np.arctan2(motor[1], motor[0])
            speed = (0.04 + 0.04 * np.clip(mag, 0, 1)) * dt * (1.0 - w.reposo_intensidad)
            w.pos = w.pos + speed * motor
        w.pos = np.clip(w.pos, -0.95, 0.95)

    def step(self, dt, peer):
        if not self.alive:
            return
        # Set peer state before brain step
        if peer is not None and peer.alive:
            self.world.peer_pos = peer.world.pos.copy()
            self.world.peer_hunger = peer.world.hunger
            self.world.peer_alert = peer.world.alerta_n3
            self.world.peer_L = peer.world.L
            self.world.peer_theta = peer.world.theta
            self.world.peer_dist = np.linalg.norm(self.world.pos - peer.world.pos)
            self.world.peer_angle = np.arctan2(peer.world.pos[1] - self.world.pos[1],
                                               peer.world.pos[0] - self.world.pos[0])
        else:
            self.world.peer_pos = None
            self.world.peer_dist = 999.0
            self.world.peer_angle = 0.0

        motor = self.brain.step(dt, self.world)
        self.apply_motor(motor, dt)

# ============================================================
# SIMULATION
# ============================================================
class Simulation:
    def __init__(self):
        self.shared = WorldState()  # Authoritative shared entity positions
        self.creature1 = Creature("Pinocho", "white", np.array([0.0, 0.0]), is_main=True)
        self.creature2 = Creature("Peer", "cyan", np.array([0.3, 0.3]), is_main=False)
        self._setup_figure()
        self._setup_entities()
        self.artists = []

    def _setup_figure(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 7))
        self.ax1.set_xlim(-1, 1)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_facecolor("black")
        self.ax1.set_title("MUNDO - 2 CRIATURAS (Pinocho blanco, Peer cyan)", color='white', fontsize=9)

        self.ax2.set_title("CORTEX DE PINOCHO - 11 AREAS CM", color='white', fontsize=9)
        self.img = self.ax2.imshow(self.creature1.brain.brain_map, vmin=0, vmax=1, cmap='inferno')

        # Region labels
        for name, reg in Brain.REGIONS.items():
            x = reg['c0'] + reg['w'] / 2
            y = reg['r0'] + 0.8
            self.ax2.text(x, y, name, color='white', fontsize=6, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor=reg['color'], alpha=0.8))

        self.info_text = self.ax2.text(0.02, 0.98, "", transform=self.ax2.transAxes, color='white',
                                       fontsize=7, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        self.peer_text = self.ax2.text(0.02, 0.55, "", transform=self.ax2.transAxes, color='cyan',
                                       fontsize=7, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def _setup_entities(self):
        # Home zones (shared)
        self.home_zone = plt.Circle((self.shared.home_pos[0], self.shared.home_pos[1]), 0.25, color='blue', alpha=0.15, fill=True)
        self.home_safety_zone = plt.Circle((self.shared.home_pos[0], self.shared.home_pos[1]), 0.30, color='blue', alpha=0.05, fill=True)
        self.ax1.add_patch(self.home_zone)
        self.ax1.add_patch(self.home_safety_zone)
        self.boundary = plt.Rectangle((-0.95, -0.95), 1.9, 1.9, edgecolor='red', linestyle='--', fill=False, alpha=0.3)
        self.ax1.add_patch(self.boundary)

        # Nest
        nido_pos = np.array([-0.70, 0.70])
        nido_tamaño = 0.30
        nido_celdas = 3
        tam_celda = nido_tamaño / nido_celdas
        inicio_x = nido_pos[0] - nido_tamaño / 2
        inicio_y = nido_pos[1] - nido_tamaño / 2
        for i in range(nido_celdas + 1):
            x = inicio_x + i * tam_celda
            self.ax1.plot([x, x], [inicio_y, inicio_y + nido_tamaño], color='yellow', linewidth=1 if i in (0, nido_celdas) else 0.5, alpha=0.8 if i in (0, nido_celdas) else 0.5)
            y = inicio_y + i * tam_celda
            self.ax1.plot([inicio_x, inicio_x + nido_tamaño], [y, y], color='yellow', linewidth=1 if i in (0, nido_celdas) else 0.5, alpha=0.8 if i in (0, nido_celdas) else 0.5)

        self.celdas = []
        for fila in range(nido_celdas):
            for col in range(nido_celdas):
                x = inicio_x + col * tam_celda
                y = inicio_y + fila * tam_celda
                celda = plt.Rectangle((x, y), tam_celda, tam_celda, facecolor='yellow', alpha=0.0, edgecolor='none')
                self.ax1.add_patch(celda)
                self.celdas.append(celda)

        # Dynamic artists - Creature 1 (Pinocho, white)
        self.dot1, = self.ax1.plot([0], [0], 'wo', markersize=7)
        self.line1, = self.ax1.plot([0, 0], [0, 0], 'w-', linewidth=2)

        # Dynamic artists - Creature 2 (Peer, cyan)
        self.dot2, = self.ax1.plot([0.3], [0.3], 'co', markersize=7)
        self.line2, = self.ax1.plot([0.3, 0.3], [0.3, 0.3], 'c-', linewidth=2)

        # Shared entities
        self.food_dot, = self.ax1.plot(self.shared.food_pos[0], self.shared.food_pos[1], 'r*', markersize=14)
        self.home_dot, = self.ax1.plot(self.shared.home_pos[0], self.shared.home_pos[1], 'bo', markersize=12)
        self.pred_dot, = self.ax1.plot(self.shared.pred_pos[0], self.shared.pred_pos[1], 'go', markersize=14)
        self.material_patch = Ellipse((0, 0), width=0.08, height=0.05, facecolor='yellow', edgecolor='yellow', alpha=0.9)
        self.ax1.add_patch(self.material_patch)
        self.material_patch.set_visible(False)

        self.artists = [self.dot1, self.line1, self.dot2, self.line2, self.img,
                        self.food_dot, self.home_dot, self.pred_dot, self.material_patch,
                        self.info_text, self.peer_text]

    def reset(self):
        self.shared = WorldState()
        self.creature1.reset(np.array([0.0, 0.0]))
        self.creature2.reset(np.array([0.3, 0.3]))
        self._actualizar_nido()

    def _actualizar_nido(self):
        # Use creature 1's deposit count as shared nest state
        celdas_llenas = min(int(np.floor(self.creature1.world.total_depositado)), MATERIALES_MAXIMOS)
        for celda in self.celdas:
            celda.set_alpha(0.0)
        for i in range(celdas_llenas):
            self.celdas[i].set_alpha(0.6)

    def _update_predator(self, dt):
        # Predator targets the nearest living creature
        candidates = []
        if self.creature1.alive:
            d1 = np.linalg.norm(self.shared.pred_pos - self.creature1.world.pos)
            candidates.append((d1, self.creature1.world.pos))
        if self.creature2.alive:
            d2 = np.linalg.norm(self.shared.pred_pos - self.creature2.world.pos)
            candidates.append((d2, self.creature2.world.pos))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0])
        nearest_dist, nearest_pos = candidates[0]

        # If nearest creature is near home, predator wanders randomly
        dist_to_home = np.linalg.norm(nearest_pos - self.shared.home_pos)
        pred_speed = 0.023 * dt
        if dist_to_home < 0.30:
            self.shared.pred_theta += np.random.uniform(-0.8, 0.8)
        else:
            to_prey = nearest_pos - self.shared.pred_pos
            self.shared.pred_theta = np.arctan2(to_prey[1], to_prey[0])

        self.shared.pred_pos = self.shared.pred_pos + pred_speed * np.array([np.cos(self.shared.pred_theta), np.sin(self.shared.pred_theta)])
        self.shared.pred_pos = np.clip(self.shared.pred_pos, -1.2, 1.2)

        # Sync to both creatures
        self.creature1.world.pred_pos = self.shared.pred_pos.copy()
        self.creature1.world.pred_theta = self.shared.pred_theta
        self.creature2.world.pred_pos = self.shared.pred_pos.copy()
        self.creature2.world.pred_theta = self.shared.pred_theta

    def update(self, frame):
        dt = TIME_WARP

        # Update shared world entities (food, home movement only)
        self.shared.step(dt)

        # Update predator toward nearest living creature
        self._update_predator(dt)

        # Update creature 1 (Pinocho)
        self.creature1.update_world(dt, self.shared)
        if self.creature1.alive:
            self.creature1.step(dt, self.creature2)

        # Update creature 2 (Peer)
        self.creature2.update_world(dt, self.shared)
        if self.creature2.alive:
            self.creature2.step(dt, self.creature1)

        # Sync shared nest state (creature 1 is authoritative for nest)
        self.creature2.world.total_depositado = self.creature1.world.total_depositado
        self.creature2.world.nido_completado = self.creature1.world.nido_completado
        self.creature2.world.material_activo = self.creature1.world.material_activo
        self.creature2.world.material_pos = self.creature1.world.material_pos.copy()

        self._actualizar_nido()
        self._update_visuals()
        return self.artists

    def _update_visuals(self):
        w1 = self.creature1.world
        w2 = self.creature2.world

        # Creature 1
        if self.creature1.alive:
            self.dot1.set_data([w1.pos[0]], [w1.pos[1]])
            head1 = w1.pos + 0.12 * np.array([np.cos(w1.theta), np.sin(w1.theta)])
            self.line1.set_data([w1.pos[0], head1[0]], [w1.pos[1], head1[1]])
        else:
            self.dot1.set_data([], [])
            self.line1.set_data([], [])

        # Creature 2
        if self.creature2.alive:
            self.dot2.set_data([w2.pos[0]], [w2.pos[1]])
            head2 = w2.pos + 0.12 * np.array([np.cos(w2.theta), np.sin(w2.theta)])
            self.line2.set_data([w2.pos[0], head2[0]], [w2.pos[1], head2[1]])
        else:
            self.dot2.set_data([], [])
            self.line2.set_data([], [])

        # Shared entities
        self.food_dot.set_data([w1.food_pos[0]], [w1.food_pos[1]])
        self.home_dot.set_data([w1.home_pos[0]], [w1.home_pos[1]])
        self.pred_dot.set_data([w1.pred_pos[0]], [w1.pred_pos[1]])
        self.home_zone.set_center((w1.home_pos[0], w1.home_pos[1]))
        self.home_safety_zone.set_center((w1.home_pos[0], w1.home_pos[1]))

        if w1.material_activo:
            self.material_patch.center = (w1.material_pos[0], w1.material_pos[1])
            self.material_patch.set_visible(True)
        else:
            self.material_patch.set_visible(False)

        # Cortex map (Pinocho's brain)
        self.img.set_data(self.creature1.brain.brain_map)
        nivel_actividad = 1.0 - _compuerta(0.25 - w1.dist_to('home'), filo=30.0) * _compuerta(0.40 - w1.hunger)
        vmax_base = 0.80
        vmax_dinamico = np.clip(vmax_base - 0.35 * nivel_actividad, 0.40, vmax_base)
        self.img.set_clim(vmin=0.0, vmax=vmax_dinamico)

        # Info text
        estado_nido = "COMPLETADO" if w1.nido_completado else f"{int(np.floor(w1.total_depositado))}/{MATERIALES_MAXIMOS}"
        n4 = self.creature1.brain.tissues['n4']
        n5 = self.creature1.brain.tissues['n5']
        dw_food_mean = np.mean([cm['dw_food'] for cm in n4.cms]) if n4.awake else 0.0
        w_emp_mean = np.mean([cm['w_empathy'] for cm in n5.cms]) if n5.awake else 0.0
        w_avoid_mean = np.mean([cm['w_avoid'] for cm in n5.cms]) if n5.awake else 0.0

        self.info_text.set_text(
            f"PIN: HAM={w1.hunger:.2f} SEG={w1.safety:.2f} ALERTA={w1.alerta_n3:.2f}\n"
            f"EDAD={w1.edad:.2f} N2={w1.instinto_construccion:.2f} N4={w1.instinto_aprendizaje:.2f} N5={w1.instinto_social:.2f}\n"
            f"CARGA={w1.L:.2f} NIDO={estado_nido}\n"
            f"N4 dw_food={dw_food_mean:+.3f}\n"
            f"N5 empathy={w_emp_mean:+.3f} avoid={w_avoid_mean:+.3f}"
        )

        # Peer text
        peer_alive = "VIVO" if self.creature2.alive else "MUERTO"
        self.peer_text.set_text(
            f"PEER: {peer_alive} HAM={w2.hunger:.2f} ALERTA={w2.alerta_n3:.2f}\n"
            f"EDAD={w2.edad:.2f} N5={w2.instinto_social:.2f}\n"
            f"DIST={w1.dist_to('peer'):.2f}"
        )

    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, interval=60, cache_frame_data=False)
        plt.tight_layout()
        plt.show(block=True)

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    sim = Simulation()
    try:
        sim.run()
    except KeyboardInterrupt:
        pass
    finally:
        sys.exit(0)
