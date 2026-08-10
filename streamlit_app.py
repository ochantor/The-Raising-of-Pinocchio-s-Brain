import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle
import time

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="🪵 The Raising of Pinocchio's Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTES (fiel al original)
# ============================================================
N = 25
ANGULOS = np.linspace(0, 2*np.pi, N, endpoint=False)
TIME_WARP = 1.0

ALPHA_MOT, T_MOT = 0.35, 0.25
ALPHA_NAV, T_NAV = 0.40, 0.15
ALPHA_N2,  T_N2  = 0.35, 0.20
ALPHA_N3,  T_N3  = 0.35, 0.20

PRED_PERCEPTION_RADIUS = 0.45
TAU_N3 = 16.0
K_ALERTA_SUBIDA = 0.9

EDAD_DESPIERTE = 0.6
RHO_MADURACION = 12.0

PICKUP_RADIUS = 0.12
NIDO_RADIUS = 0.25
K_PICKUP = 0.55
K_DROP = 0.45
STEEPNESS_GATE = 50.0

IMPULSO_MINIMO = 0.15

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def _compuerta(x, filo=30.0):
    z = np.clip(-filo * x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))

def relajar_softmax(actividad, energias, alpha, temperatura):
    e = energias - np.max(energias)
    pesos = np.exp(e / temperatura)
    objetivo = pesos / np.sum(pesos)
    return (1.0 - alpha) * actividad + alpha * objetivo

def _blur5(v):
    p = np.pad(v, 1, mode='constant')
    k = np.array([[0.06, 0.12, 0.06],
                  [0.12, 0.28, 0.12],
                  [0.06, 0.12, 0.06]])
    out = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            out[i, j] = np.sum(p[i:i+3, j:j+3] * k)
    return out

def generar_material(materiales_generados, max_materiales_generados, nido_completado,
                     nido_pos, home_pos):
    if materiales_generados >= max_materiales_generados or nido_completado:
        return np.array([0.0, 0.0]), False
    centro_x, centro_y, radio = 0.75, -0.75, 0.20
    for _ in range(200):
        angulo = np.random.uniform(0, 2*np.pi)
        distancia = np.random.uniform(0, radio)
        nueva_pos = np.array([centro_x + distancia*np.cos(angulo),
                              centro_y + distancia*np.sin(angulo)])
        if (np.linalg.norm(nueva_pos - nido_pos) > 0.20 and
            np.linalg.norm(nueva_pos - home_pos) > 0.20):
            return nueva_pos, True
    return np.array([0.0, 0.0]), False

# ============================================================
# INICIALIZACIÓN DEL ESTADO
# ============================================================
def init_state():
    # CMS
    cms_mot = []
    for k in range(N):
        cms_mot.append({
            "angle": ANGULOS[k],
            "food_weight": np.random.uniform(1.2, 1.6),
            "home_weight": np.random.uniform(1.3, 1.7),
            "pred_weight": np.random.uniform(1.2, 1.7),
            "explore": np.random.uniform(0, 0.06)
        })
    cms_nav = []
    for k in range(N):
        cms_nav.append({
            "angle": ANGULOS[k],
            "border_weight": np.random.uniform(0.8, 1.2),
            "explore": np.random.uniform(0, 0.05)
        })
    cms_n2 = []
    for k in range(N):
        cms_n2.append({
            "angle": ANGULOS[k],
            "material_weight": np.random.uniform(1.2, 1.6),
            "nido_weight": np.random.uniform(1.3, 1.7),
            "explore": np.random.uniform(0, 0.04)
        })
    cms_n3 = []
    for k in range(N):
        cms_n3.append({
            "angle": ANGULOS[k],
            "escape_weight": np.random.uniform(1.2, 1.7),
            "explore": np.random.uniform(0, 0.04)
        })

    food_theta = np.random.uniform(0, 2*np.pi)
    home_theta = np.random.uniform(0, 2*np.pi)
    pred_theta = np.random.uniform(0, 2*np.pi)
    food_radius = 0.65
    home_radius = 0.60
    pred_radius = 0.8

    return {
        'pos': np.array([0.0, 0.0]),
        'theta': 0.0,
        'hunger': 0.3,
        'safety': 0.0,
        'danger': 0.0,
        'border_stress': 0.0,
        'alerta_n3': 0.0,
        'food_pos': np.array([food_radius*np.cos(food_theta), 0.55*np.sin(1.7*food_theta)]),
        'home_pos': np.array([home_radius*np.cos(home_theta), 0.45*np.sin(1.3*home_theta)]),
        'pred_pos': np.array([pred_radius*np.sin(pred_theta), pred_radius*np.cos(pred_theta)]),
        'food_theta': food_theta,
        'home_theta': home_theta,
        'pred_theta': pred_theta,
        'nido_pos': np.array([-0.70, 0.70]),
        'nido_tamaño': 0.30,
        'nido_celdas': 3,
        'materiales_maximos': 9,
        'nido_completado': False,
        'total_depositado': 0.0,
        'total_recogido_frac': 0.0,
        'material_pos': np.array([0.0, 0.0]),
        'material_activo': False,
        'L': 0.0,
        'edad': 0.0,
        'instinto_construccion': 0.0,
        'impulso_construir': 0.8,
        'tasa_impulso_base': 0.015,
        'urgencia_constructiva': 1.2,
        'materiales_generados': 0,
        'max_materiales_generados': 30,
        'tiempo_sin_material': 0.0,
        'tiempo_sin_construir': 0.0,
        'food_lock': False,
        'home_lock': False,
        'activity_mot': np.ones(N) / N,
        'activity_nav': np.ones(N) / N,
        'activity_n2': np.ones(N) / N,
        'activity_n3': np.ones(N) / N,
        'cms_mot': cms_mot,
        'cms_nav': cms_nav,
        'cms_n2': cms_n2,
        'cms_n3': cms_n3,
        'brain': np.zeros((21, 21)),
        'step_count': 0,
        'alive': True,
    }

if 'sim' not in st.session_state:
    st.session_state.sim = init_state()

sim = st.session_state.sim

# ============================================================
# PASO DE SIMULACIÓN
# ============================================================
def step_simulation():
    s = st.session_state.sim
    dt = TIME_WARP

    if not s['alive']:
        return

    # --- distancias ---
    dist_food = np.linalg.norm(s['pos'] - s['food_pos'])
    dist_home = np.linalg.norm(s['pos'] - s['home_pos'])
    dist_pred = np.linalg.norm(s['pos'] - s['pred_pos'])
    dist_nido = np.linalg.norm(s['pos'] - s['nido_pos'])
    dist_material = np.linalg.norm(s['pos'] - s['material_pos']) if s['material_activo'] else 999.0

    # --- homeostasis ---
    s['hunger'] = np.clip(s['hunger'] + 0.0015*dt, 0, 1)
    s['safety'] = np.clip(s['safety'] + (0.0 if dist_home < 0.25 else 0.0020*dt) - (s['safety'] if dist_home < 0.25 else 0.0), 0, 1)
    if dist_home < 0.25:
        s['safety'] = 0.0
    s['danger'] = np.clip(s['danger'] + 0.003*dt, 0, 1)

    # --- gating homeostatico continuo ---
    compuerta_en_casa = _compuerta(0.25 - dist_home)
    compuerta_poco_hambre = _compuerta(0.40 - s['hunger'])
    compuerta_seguridad_alta = _compuerta(s['safety'] - 0.80)
    compuerta_urgencia = _compuerta(s['hunger'] - 0.50) * _compuerta(0.40 - s['safety'])

    atenuacion = np.clip((1.0 - s['safety']) / 0.20, 0, 1)
    eh_atenuado = s['hunger'] * atenuacion
    sn_atenuado = s['safety'] * 1.8
    eh_urgente = s['hunger'] * 1.5
    sn_urgente = s['safety'] * 0.3
    eh_plano = s['hunger']
    sn_plano = s['safety']

    peso_atenuado = compuerta_seguridad_alta
    peso_urgente = compuerta_urgencia * (1.0 - peso_atenuado)
    peso_plano = np.clip(1.0 - peso_atenuado - peso_urgente, 0, 1)

    effective_hunger_lejos = peso_atenuado*eh_atenuado + peso_urgente*eh_urgente + peso_plano*eh_plano
    current_safety_need_lejos = peso_atenuado*sn_atenuado + peso_urgente*sn_urgente + peso_plano*sn_plano

    en_casa_activo = compuerta_en_casa * (1.0 - compuerta_poco_hambre)
    reposo_intensidad = compuerta_en_casa * compuerta_poco_hambre
    lejos = 1.0 - compuerta_en_casa

    effective_hunger = en_casa_activo*s['hunger'] + lejos*effective_hunger_lejos
    current_safety_need = reposo_intensidad*1.0 + lejos*current_safety_need_lejos

    # --- edad e instinto ---
    s['edad'] += 0.01 * dt
    s['instinto_construccion'] = _compuerta(s['edad'] - EDAD_DESPIERTE, filo=RHO_MADURACION)

    # --- urgencia constructiva ---
    if not s['nido_completado']:
        s['urgencia_constructiva'] += 0.005*dt
        s['urgencia_constructiva'] += 0.015*dt * _compuerta(dist_material - 999 + 1) * (1.0 - s['L'])
        s['urgencia_constructiva'] += 0.02*dt * s['L']
        progreso = s['total_depositado'] / s['materiales_maximos']
        s['urgencia_constructiva'] += 0.005*progreso*dt
        s['tiempo_sin_construir'] += dt
        s['urgencia_constructiva'] += 0.01*dt * _compuerta(s['tiempo_sin_construir'] - 60, filo=0.3)
        s['urgencia_constructiva'] = np.clip(s['urgencia_constructiva'], 0, 1.5)

    # --- estres de bordes ---
    dist_to_wall_x = 1.0 - abs(s['pos'][0])
    dist_to_wall_y = 1.0 - abs(s['pos'][1])
    closest_wall_dist = min(dist_to_wall_x, dist_to_wall_y)
    s['border_stress'] = (np.clip((0.25 - closest_wall_dist)/0.25, 0, 1) ** 2) if closest_wall_dist < 0.25 else 0.0

    # --- movimiento de entidades ---
    s['food_theta'] += 0.015*dt
    s['food_pos'] = np.array([0.65*np.cos(s['food_theta']), 0.55*np.sin(1.7*s['food_theta'])])
    s['home_theta'] -= 0.010*dt
    s['home_pos'] = np.array([0.60*np.cos(s['home_theta']), 0.45*np.sin(1.3*s['home_theta'])])

    # --- depredador ---
    pred_speed = 0.023*dt
    if dist_home < 0.30:
        s['pred_theta'] += np.random.uniform(-0.8, 0.8)
    else:
        to_prey = s['pos'] - s['pred_pos']
        s['pred_theta'] = np.arctan2(to_prey[1], to_prey[0])
    s['pred_pos'] = s['pred_pos'] + pred_speed*np.array([np.cos(s['pred_theta']), np.sin(s['pred_theta'])])
    s['pred_pos'] = np.clip(s['pred_pos'], -1.2, 1.2)

    # --- N+3: histeresis ---
    percepcion_amenaza = _compuerta(PRED_PERCEPTION_RADIUS - dist_pred, filo=25.0)
    subida_alerta = K_ALERTA_SUBIDA * percepcion_amenaza * (1.0 - s['alerta_n3'])
    bajada_alerta = (s['alerta_n3'] / TAU_N3) * (1.0 - percepcion_amenaza)
    s['alerta_n3'] = np.clip(s['alerta_n3'] + dt*(subida_alerta - bajada_alerta), 0, 1)

    # --- colisiones ---
    if dist_food < 0.10 and not s['food_lock']:
        s['hunger'] = 0.0
        s['food_lock'] = True
    if dist_home < 0.10 and not s['home_lock']:
        s['home_lock'] = True
    if dist_pred < 0.12:
        if dist_home < 0.30:
            s['danger'] = 0.0
        else:
            st.session_state.sim = init_state()
            st.session_state.sim['step_count'] = s['step_count'] + 1
            return
    if dist_food > 0.18: s['food_lock'] = False
    if dist_home > 0.18: s['home_lock'] = False

    # --- impulso constructivo ---
    if s['nido_completado']:
        s['impulso_construir'] *= 0.95
    else:
        s['impulso_construir'] += s['tasa_impulso_base']*dt
        s['impulso_construir'] += s['urgencia_constructiva']*0.02*dt
        s['impulso_construir'] += 0.01*dt * (1.0 - s['L']) * (1.0 if s['material_activo'] else 0.0)
        s['impulso_construir'] += 0.02*dt * s['L']
        progreso = s['total_depositado'] / s['materiales_maximos']
        s['impulso_construir'] += 0.005*progreso*dt
        s['impulso_construir'] += 0.01*dt * _compuerta(s['tiempo_sin_construir'] - 50, filo=0.3)
    s['impulso_construir'] = np.clip(s['impulso_construir'], 0, 1)

    # --- carga continua L ---
    g_pickup = _compuerta(PICKUP_RADIUS - dist_material, filo=STEEPNESS_GATE) if s['material_activo'] else 0.0
    g_drop = _compuerta(NIDO_RADIUS - dist_nido, filo=STEEPNESS_GATE) if not s['nido_completado'] else 0.0

    flujo_entrada = K_PICKUP * g_pickup * (1.0 - s['L'])
    flujo_salida = K_DROP * g_drop * s['L']
    s['L'] = np.clip(s['L'] + dt*(flujo_entrada - flujo_salida), 0, 1)

    s['total_depositado'] = min(s['materiales_maximos'], s['total_depositado'] + dt*flujo_salida)
    s['total_recogido_frac'] += dt*flujo_entrada
    if s['total_recogido_frac'] >= 1.0 and s['material_activo']:
        s['material_activo'] = False
        s['total_recogido_frac'] -= 1.0

    if s['total_depositado'] >= s['materiales_maximos']:
        s['nido_completado'] = True
        s['material_activo'] = False

    # --- generar nuevo material ---
    if not s['nido_completado'] and not s['material_activo']:
        if s['materiales_generados'] < s['max_materiales_generados']:
            mat_pos, activo = generar_material(
                s['materiales_generados'], s['max_materiales_generados'],
                s['nido_completado'], s['nido_pos'], s['home_pos']
            )
            s['material_pos'] = mat_pos
            s['material_activo'] = activo
            if activo:
                s['materiales_generados'] += 1
        else:
            s['tiempo_sin_material'] += dt
            if s['tiempo_sin_material'] > 150:
                s['materiales_generados'] = 0
                s['tiempo_sin_material'] = 0.0
                mat_pos, activo = generar_material(
                    s['materiales_generados'], s['max_materiales_generados'],
                    s['nido_completado'], s['nido_pos'], s['home_pos']
                )
                s['material_pos'] = mat_pos
                s['material_activo'] = activo
                if activo:
                    s['materiales_generados'] += 1

    # --- direcciones sensoriales ---
    vec_food = s['food_pos'] - s['pos']
    vec_home = s['home_pos'] - s['pos']
    vec_pred = s['pred_pos'] - s['pos']
    angle_food = np.arctan2(vec_food[1], vec_food[0])
    angle_home = np.arctan2(vec_home[1], vec_home[0])
    angle_pred = np.arctan2(vec_pred[1], vec_pred[0])

    fuerza_oeste = 1.0/(1.0 + (s['pos'][0]-(-0.95)))
    fuerza_este = 1.0/(1.0 + (0.95-s['pos'][0]))
    fuerza_sur = 1.0/(1.0 + (s['pos'][1]-(-0.95)))
    fuerza_norte = 1.0/(1.0 + (0.95-s['pos'][1]))
    vec_border = np.array([fuerza_este-fuerza_oeste, fuerza_norte-fuerza_sur])
    angle_border = np.arctan2(vec_border[1], vec_border[0])

    # --- AREA MOT ---
    current_danger_factor = 0.0 if dist_home < 0.25 else s['alerta_n3']
    angs = np.array([cm["angle"] for cm in s['cms_mot']])
    fw = np.array([cm["food_weight"] for cm in s['cms_mot']])
    hw = np.array([cm["home_weight"] for cm in s['cms_mot']])
    pw = np.array([cm["pred_weight"] for cm in s['cms_mot']])
    ex = np.array([cm["explore"] for cm in s['cms_mot']])
    energies_mot = (effective_hunger*fw*np.cos(angle_food-angs)
                    + current_safety_need*hw*np.cos(angle_home-angs)
                    - current_danger_factor*pw*np.cos(angle_pred-angs)
                    + ex*np.random.uniform(-1, 1, N)
                    + 0.06*np.random.randn(N))
    s['activity_mot'] = relajar_softmax(s['activity_mot'], energies_mot, ALPHA_MOT, T_MOT)

    # --- AREA NAV ---
    angs_n = np.array([cm["angle"] for cm in s['cms_nav']])
    bw = np.array([cm["border_weight"] for cm in s['cms_nav']])
    exn = np.array([cm["explore"] for cm in s['cms_nav']])
    energies_nav = -s['border_stress']*bw*np.cos(angle_border-angs_n) + exn*np.random.uniform(-1, 1, N)
    ruido_reposo = np.random.uniform(-1, 1, N) * 2.0
    energies_nav_eff = (1.0 - reposo_intensidad)*energies_nav + reposo_intensidad*ruido_reposo
    s['activity_nav'] = relajar_softmax(s['activity_nav'], energies_nav_eff, ALPHA_NAV, T_NAV)

    # --- AREA N+2 ---
    vec_material = (s['material_pos'] - s['pos']) if s['material_activo'] else np.array([0.0, 0.0])
    vec_nido = s['nido_pos'] - s['pos']
    angle_material = np.arctan2(vec_material[1], vec_material[0]) if np.linalg.norm(vec_material) > 0 else 0.0
    angle_nido = np.arctan2(vec_nido[1], vec_nido[0])
    material_disponible = 1.0 if s['material_activo'] else 0.0

    angs2 = np.array([cm["angle"] for cm in s['cms_n2']])
    mw = np.array([cm["material_weight"] for cm in s['cms_n2']])
    nw = np.array([cm["nido_weight"] for cm in s['cms_n2']])
    ex2 = np.array([cm["explore"] for cm in s['cms_n2']])
    material_align = np.cos(angle_material-angs2) if s['material_activo'] else np.zeros(N)
    nido_align = np.cos(angle_nido-angs2)

    energies_n2 = (
        (1-s['L'])*material_disponible*mw*material_align*(0.8+0.5*s['instinto_construccion'])
        + s['L']*nw*nido_align*(0.7+0.5*s['instinto_construccion'])
        + s['L']*mw*material_align*0.1
        + (1-s['L'])*material_disponible*nw*nido_align*0.1
        + (1-s['L'])*(1-material_disponible)*ex2*np.random.uniform(0.5, 1.5, N)*(0.3+s['instinto_construccion'])
        + ex2*np.random.uniform(-1, 1, N)
    )
    s['activity_n2'] = relajar_softmax(s['activity_n2'], energies_n2, ALPHA_N2, T_N2)

    # --- AREA N+3 ---
    angle_evasion = angle_pred + np.pi
    angs3 = np.array([cm["angle"] for cm in s['cms_n3']])
    ew = np.array([cm["escape_weight"] for cm in s['cms_n3']])
    ex3 = np.array([cm["explore"] for cm in s['cms_n3']])
    evasion_align = np.cos(angle_evasion - angs3)
    energies_n3 = (
        s['alerta_n3'] * ew * evasion_align
        + ex3 * np.random.uniform(-1, 1, N)
        + 0.06 * np.random.randn(N)
    )
    s['activity_n3'] = relajar_softmax(s['activity_n3'], energies_n3, ALPHA_N3, T_N3)

    # --- SINTESIS MOTORA ---
    vecs = np.stack([np.cos(ANGULOS), np.sin(ANGULOS)], axis=1)
    motor_mot = s['activity_mot'] @ vecs
    motor_nav = s['activity_nav'] @ vecs
    motor_n2 = s['activity_n2'] @ vecs
    motor_n3 = s['activity_n3'] @ vecs
    if np.linalg.norm(motor_mot) > 0: motor_mot = motor_mot/np.linalg.norm(motor_mot)
    if np.linalg.norm(motor_nav) > 0: motor_nav = motor_nav/np.linalg.norm(motor_nav)
    if np.linalg.norm(motor_n2) > 0: motor_n2 = motor_n2/np.linalg.norm(motor_n2)
    if np.linalg.norm(motor_n3) > 0: motor_n3 = motor_n3/np.linalg.norm(motor_n3)

    # --- INTEGRACION ---
    g_impulso = _compuerta(s['impulso_construir'] - IMPULSO_MINIMO, filo=40.0)
    peso_n2 = s['instinto_construccion'] * g_impulso * s['impulso_construir'] * 0.7
    factor_hambre = 1.0 - s['hunger']*0.5
    peso_n2 *= factor_hambre
    factor_peligro = np.clip(dist_pred/0.4, 0, 1)
    peso_n2 *= factor_peligro
    peso_n2 *= (1.0 + 0.3*s['L'])
    peso_n2 = np.clip(peso_n2, 0, 0.7)

    peso_n3 = np.clip(s['alerta_n3'] * 0.65, 0, 0.65)
    suma_pesos = peso_n2 + peso_n3
    if suma_pesos > 1.0:
        factor_norm = 1.0 / suma_pesos
        peso_n2 *= factor_norm
        peso_n3 *= factor_norm

    motor_primario = (1.0-peso_n2-peso_n3)*motor_mot + peso_n2*motor_n2 + peso_n3*motor_n3
    motor = (1.0-s['border_stress'])*motor_primario + s['border_stress']*motor_nav
    if np.linalg.norm(motor) > 0:
        motor = motor/np.linalg.norm(motor)

    # --- movimiento ---
    mag = np.linalg.norm(motor)
    if mag > 0:
        s['theta'] = np.arctan2(motor[1], motor[0])
        speed = (0.04 + 0.04*np.clip(mag, 0, 1)) * dt * (1.0 - reposo_intensidad)
        s['pos'] = s['pos'] + speed*motor
    s['pos'] = np.clip(s['pos'], -0.95, 0.95)

    # --- MAPA CORTICAL ---
    cm = _blur5(s['activity_mot'][:25].reshape(5, 5))
    cn = _blur5(s['activity_nav'][:25].reshape(5, 5))
    c2 = _blur5(s['activity_n2'][:25].reshape(5, 5))
    c3 = _blur5(s['activity_n3'][:25].reshape(5, 5))

    reposo = compuerta_en_casa
    decay_activo = 0.84
    decay_reposo = 0.70
    s['brain'] *= (decay_activo - (decay_activo - decay_reposo) * reposo)
    gain = 0.50 * (1.0 - 0.45 * reposo)
    noise_amp = 0.03
    nm = np.random.rand(5, 5) * noise_amp
    nn = np.random.rand(5, 5) * noise_amp
    n2 = np.random.rand(5, 5) * noise_amp
    n3 = np.random.rand(5, 5) * noise_amp

    s['brain'][7:12, 5:10] += cm * gain + nm
    s['brain'][7:12, 12:17] += cn * gain + nn
    s['brain'][13:18, 12:17] += c2 * gain + n2
    s['brain'][13:18, 5:10] += c3 * gain * (0.3 + 0.7*s['alerta_n3']) + n3

    bias = 0.03 * reposo
    s['brain'][7:12, 5:10] += bias
    s['brain'][7:12, 12:17] += bias
    s['brain'][13:18, 12:17] += bias
    s['brain'][13:18, 5:10] += bias

    s['brain'] = np.clip(s['brain'], 0, 0.80)
    s['step_count'] += 1

# ============================================================
# RENDERIZADO
# ============================================================
def render_world(s):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_title("MUNDO — Competencia Softmax Continua + N+3 (Histeresis)", color='white', fontsize=10)

    # Zonas
    home_zone = Circle((s['home_pos'][0], s['home_pos'][1]), 0.25, color='blue', alpha=0.15)
    ax.add_patch(home_zone)
    home_safety = Circle((s['home_pos'][0], s['home_pos'][1]), 0.30, color='blue', alpha=0.05)
    ax.add_patch(home_safety)
    boundary = Rectangle((-0.95, -0.95), 1.9, 1.9, edgecolor='red', linestyle='--', fill=False, alpha=0.3)
    ax.add_patch(boundary)

    # Nido
    inicio_x = s['nido_pos'][0] - s['nido_tamaño'] / 2
    inicio_y = s['nido_pos'][1] - s['nido_tamaño'] / 2
    tam_celda = s['nido_tamaño'] / s['nido_celdas']
    ax.plot([inicio_x, inicio_x + s['nido_tamaño']], [inicio_y, inicio_y], color='yellow', linewidth=2, alpha=0.8)
    ax.plot([inicio_x, inicio_x + s['nido_tamaño']], [inicio_y + s['nido_tamaño'], inicio_y + s['nido_tamaño']], color='yellow', linewidth=2, alpha=0.8)
    ax.plot([inicio_x, inicio_x], [inicio_y, inicio_y + s['nido_tamaño']], color='yellow', linewidth=2, alpha=0.8)
    ax.plot([inicio_x + s['nido_tamaño'], inicio_x + s['nido_tamaño']], [inicio_y, inicio_y + s['nido_tamaño']], color='yellow', linewidth=2, alpha=0.8)
    for i in range(1, s['nido_celdas']):
        x = inicio_x + i * tam_celda
        ax.plot([x, x], [inicio_y, inicio_y + s['nido_tamaño']], color='yellow', linewidth=1, alpha=0.5)
        y = inicio_y + i * tam_celda
        ax.plot([inicio_x, inicio_x + s['nido_tamaño']], [y, y], color='yellow', linewidth=1, alpha=0.5)

    celdas_llenas = min(int(np.floor(s['total_depositado'])), s['materiales_maximos'])
    idx = 0
    for fila in range(3):
        for col in range(3):
            x = inicio_x + col * tam_celda
            y = inicio_y + fila * tam_celda
            if idx < celdas_llenas:
                cell = Rectangle((x, y), tam_celda, tam_celda, facecolor='yellow', alpha=0.6, edgecolor='none')
                ax.add_patch(cell)
            idx += 1

    # Entidades
    ax.plot(s['food_pos'][0], s['food_pos'][1], 'r*', markersize=16, label='Comida')
    ax.plot(s['home_pos'][0], s['home_pos'][1], 'bo', markersize=12, label='Casa')
    ax.plot(s['pred_pos'][0], s['pred_pos'][1], 'go', markersize=14, label='Depredador')

    if s['material_activo']:
        mat = Ellipse((s['material_pos'][0], s['material_pos'][1]), width=0.08, height=0.05,
                      facecolor='yellow', edgecolor='yellow', alpha=0.9)
        ax.add_patch(mat)

    # Pinocchio
    ax.plot(s['pos'][0], s['pos'][1], 'wo', markersize=8)
    head = s['pos'] + 0.12*np.array([np.cos(s['theta']), np.sin(s['theta'])])
    ax.plot([s['pos'][0], head[0]], [s['pos'][1], head[1]], 'w-', linewidth=2)

    ax.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
    return fig

def render_brain(s):
    fig, ax = plt.subplots(figsize=(7, 7))

    nivel_actividad = 1.0 - _compuerta(0.25 - np.linalg.norm(s['pos'] - s['home_pos']))
    vmax_base = 0.80
    vmax_dinamico = vmax_base - 0.35 * nivel_actividad
    vmax_dinamico = np.clip(vmax_dinamico, 0.40, vmax_base)

    im = ax.imshow(s['brain'], vmin=0, vmax=vmax_dinamico, cmap='inferno', interpolation='nearest')
    ax.axvline(10.5, color='white', linestyle=':', alpha=0.2, linewidth=0.5)
    ax.axhline(12, color='white', linestyle=':', alpha=0.2, linewidth=0.5)

    ax.text(7, 12.4, 'MOT', color='white', fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8))
    ax.text(14, 12.4, 'NAV', color='white', fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.8))
    ax.text(7, 18.4, 'N+3', color='white', fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='darkorange', alpha=0.8))
    ax.text(14, 18.4, 'N+2', color='white', fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))

    ax.set_title("CORTEZA — Competencia Softmax Continua", color='white', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    estado_nido = "COMPLETADO" if s['nido_completado'] else f"{int(np.floor(s['total_depositado']))}/{s['materiales_maximos']}"
    info = (
        f"HAMBRE: {s['hunger']:.2f}  |  SEGURIDAD: {s['safety']:.2f}\n"
        f"IMPULSO: {s['impulso_construir']:.2f}  |  URGENCIA: {s['urgencia_constructiva']:.2f}\n"
        f"CARGA L: {s['L']:.2f}  |  N+2 μ: {np.mean(s['activity_n2']):.3f}\n"
        f"NIDO: {estado_nido}  |  EDAD: {s['edad']:.2f}  |  INSTINTO: {s['instinto_construccion']:.2f}\n"
        f"⚡ ALERTA N+3: {s['alerta_n3']:.2f}  (τ={TAU_N3:.0f})"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, color='white', fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    return fig

# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.title("🪵✨ The Raising of Pinocchio's Brain")
st.markdown("*25 Microcircuitos Corticales compitiendo vía softmax · Sin argmax, sin booleanos*")

# Sidebar
with st.sidebar:
    st.header("🎮 Controles")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏭️ Paso", use_container_width=True):
            step_simulation()
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sim = init_state()
            st.rerun()
    with col3:
        if st.button("▶️ Auto 10", use_container_width=True):
            for _ in range(10):
                step_simulation()

    auto_run = st.toggle("▶️ Auto-run continuo", value=False)
    speed_ms = st.slider("Velocidad (ms)", 50, 500, 150)

    st.divider()
    st.markdown("**📊 Estado Actual**")
    s = st.session_state.sim
    st.metric("⚡ Alerta N+3", f"{s['alerta_n3']:.3f}")
    st.metric("🍖 Hambre", f"{s['hunger']:.2f}")
    st.metric("🏠 Seguridad", f"{s['safety']:.2f}")
    st.metric("📦 Carga L", f"{s['L']:.2f}")
    st.metric("🦶 Paso", s['step_count'])
    st.metric("📐 θ", f"{np.degrees(s['theta']):.1f}°")

# Panel principal
s = st.session_state.sim
col_w, col_b = st.columns(2)

with col_w:
    st.subheader("🌍 El Mundo")
    fig_w = render_world(s)
    st.pyplot(fig_w, use_container_width=True)

with col_b:
    st.subheader("🧠 Mapa Cortical 21×21")
    fig_b = render_brain(s)
    st.pyplot(fig_b, use_container_width=True)

# Auto-run
if auto_run:
    time.sleep(speed_ms / 1000)
    step_simulation()
    st.rerun()
