#!/usr/bin/env python3
"""
Ant Colony Simulation — PySide6 + NumPy + Numba
Features 6 Algorithms: ACO, Markov Chain, MDP, Q-Learning, Lévy Flight, Genetic Algorithm.

Controls:
  Left-click        → place a new food source
  Right-click/drag  → draw obstacle walls
  Space             → pause / resume
  R                 → reset simulation
  P                 → toggle pheromone visibility
  C                 → clear all obstacles
"""

import sys
import math
import time
from collections import deque

import numpy as np
from numba import njit

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QSizePolicy,
)
from PySide6.QtCore import QTimer, Qt, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QImage, QRadialGradient, QFont, QPolygonF,
)

# ── World ────────────────────────────────────────────────────
WORLD_W, WORLD_H = 960, 640
CELL = 4
GW = WORLD_W // CELL
GH = WORLD_H // CELL

NEST_X, NEST_Y = WORLD_W // 2, WORLD_H // 2
NEST_R = 24

# ── Ants ─────────────────────────────────────────────────────
MAX_ANTS    = 2000
DEFAULT_ANTS = 500
SENSOR_DIST = 20.0

# ── Pheromones ───────────────────────────────────────────────
MAX_PHER           = 500.0
NEST_PHER_RADIUS   = 60.0
NEST_PHER_STRENGTH = 80.0

# ── Food ─────────────────────────────────────────────────────
FOOD_R      = 12
FOOD_AMOUNT = 500
MAX_FOODS   = 100

# ── Obstacles ────────────────────────────────────────────────
OBS_RADIUS = 14

# ── Algorithm IDs ────────────────────────────────────────────
ALGO_ACO     = 0
ALGO_MARKOV  = 1
ALGO_MDP     = 2
ALGO_QLEARN  = 3
ALGO_LEVY    = 4
ALGO_GENETIC = 5


# ══════════════════════════════════════════════════════════════
# NUMBA JIT COMPILED FUNCTIONS
# ══════════════════════════════════════════════════════════════

@njit(cache=True)
def sense_pheromone(x, y, angle, grid, cell, dist, gw, gh, obs_grid):
    sx = x + math.cos(angle) * dist
    sy = y + math.sin(angle) * dist
    gx = int(sx / cell)
    gy = int(sy / cell)
    if 0 <= gx < gw and 0 <= gy < gh:
        if obs_grid[gy, gx]:
            return 0.0
        return grid[gy, gx]
    return 0.0


@njit(cache=True)
def step_ants(ant_x, ant_y, ant_angle, ant_has_food, ant_deposit, ant_levy_timer,
              ant_genes, ant_fitness,
              food_pher, home_pher, food_x, food_y, food_amount,
              n_foods, collected_count, obs_grid, visit_grid,
              V_food, V_nest, Q_table,
              nest_x, nest_y, nest_r_sq, food_r_sq,
              world_w, world_h, cell, gw, gh,
              sensor_dist, turn_rate, wander_noise,
              ant_speed, max_pher, pher_deposit, n_active, algo_id):
    
    for i in range(n_active):
        x = ant_x[i]
        y = ant_y[i]
        angle = ant_angle[i]
        has_food = ant_has_food[i]
        deposit = ant_deposit[i]
        action = 0 # For Q-Learning

        # Track visits for Markov Stationary Distribution
        gx_cur = int(x / cell)
        gy_cur = int(y / cell)
        if 0 <= gx_cur < gw and 0 <= gy_cur < gh:
            visit_grid[gy_cur, gx_cur] += 1.0

        # --- Algorithm Selection ---
        if algo_id == ALGO_ACO or algo_id == ALGO_MARKOV or algo_id == ALGO_GENETIC:
            t_r = turn_rate
            s_a = 0.7
            w_n = wander_noise
            
            # Genetic algorithm uses evolved genes
            if algo_id == ALGO_GENETIC:
                t_r = ant_genes[i, 0]
                s_a = ant_genes[i, 1]
                w_n = ant_genes[i, 2]

            if has_food:
                s_l = sense_pheromone(x, y, angle - s_a, home_pher, cell, sensor_dist, gw, gh, obs_grid)
                s_c = sense_pheromone(x, y, angle,             home_pher, cell, sensor_dist, gw, gh, obs_grid)
                s_r = sense_pheromone(x, y, angle + s_a, home_pher, cell, sensor_dist, gw, gh, obs_grid)
            else:
                s_l = sense_pheromone(x, y, angle - s_a, food_pher, cell, sensor_dist, gw, gh, obs_grid)
                s_c = sense_pheromone(x, y, angle,             food_pher, cell, sensor_dist, gw, gh, obs_grid)
                s_r = sense_pheromone(x, y, angle + s_a, food_pher, cell, sensor_dist, gw, gh, obs_grid)

            # Markov uses stochastic transition, others use deterministic comparison
            if algo_id == ALGO_MARKOV:
                total = s_l + s_c + s_r + 1e-6
                p_l = s_l / total; p_c = s_c / total; p_r = s_r / total
                rand = np.random.random()
                if rand < p_l: angle -= t_r * (0.5 + np.random.random())
                elif rand < p_l + p_c: pass
                else: angle += t_r * (0.5 + np.random.random())
                angle += w_n * (2.0 * np.random.random() - 1.0)
            else:
                if s_c >= s_l and s_c >= s_r: pass
                elif s_l > s_r: angle -= t_r * (0.4 + np.random.random())
                elif s_r > s_l: angle += t_r * (0.4 + np.random.random())
                else: angle += t_r * (2.0 * np.random.random() - 1.0)
                angle += w_n * (2.0 * np.random.random() - 1.0)

        elif algo_id == ALGO_MDP:
            # Follow the gradient of the computed optimal Value Function
            if has_food:
                s_l = sense_pheromone(x, y, angle - 0.7, V_nest, cell, sensor_dist, gw, gh, obs_grid)
                s_c = sense_pheromone(x, y, angle,             V_nest, cell, sensor_dist, gw, gh, obs_grid)
                s_r = sense_pheromone(x, y, angle + 0.7, V_nest, cell, sensor_dist, gw, gh, obs_grid)
            else:
                s_l = sense_pheromone(x, y, angle - 0.7, V_food, cell, sensor_dist, gw, gh, obs_grid)
                s_c = sense_pheromone(x, y, angle,             V_food, cell, sensor_dist, gw, gh, obs_grid)
                s_r = sense_pheromone(x, y, angle + 0.7, V_food, cell, sensor_dist, gw, gh, obs_grid)

            if s_c >= s_l and s_c >= s_r: pass
            elif s_l > s_r: angle -= turn_rate * 0.8
            else: angle += turn_rate * 0.8
            angle += wander_noise * 0.5 * (2.0 * np.random.random() - 1.0)

        elif algo_id == ALGO_QLEARN:
            # Epsilon-Greedy Q-Learning
            if 0 <= gx_cur < gw and 0 <= gy_cur < gh:
                rand_val = np.random.random()
                if rand_val < 0.15: # Exploration
                    action = np.random.randint(0, 3)
                else: # Exploitation
                    action = 0
                    if Q_table[gy_cur, gx_cur, 1] > Q_table[gy_cur, gx_cur, 0]: action = 1
                    if Q_table[gy_cur, gx_cur, 2] > Q_table[gy_cur, gx_cur, action]: action = 2
                
                if action == 0: angle -= turn_rate
                elif action == 2: angle += turn_rate
            angle += wander_noise * (2.0 * np.random.random() - 1.0)

        elif algo_id == ALGO_LEVY:
            # Lévy Flight: Long straight walks interspersed with sharp turns
            if ant_levy_timer[i] > 0:
                ant_levy_timer[i] -= 1
                # Slight pheromone steering while walking straight
                if has_food: s_c = sense_pheromone(x, y, angle, home_pher, cell, sensor_dist, gw, gh, obs_grid)
                else: s_c = sense_pheromone(x, y, angle, food_pher, cell, sensor_dist, gw, gh, obs_grid)
                if s_c < 0.5: angle += 0.2 * (2.0 * np.random.random() - 1.0)
            else:
                angle += np.random.random() * 2 * math.pi
                jump_len = int(1.0 / (np.random.random() + 1e-6))
                ant_levy_timer[i] = min(jump_len, 60)

        # Move forward
        old_x, old_y = x, y
        x += math.cos(angle) * ant_speed
        y += math.sin(angle) * ant_speed

        # Bounce off walls
        margin = 4.0
        if x < margin: x = margin; angle = math.pi - angle + 0.6 * np.random.random() - 0.3
        elif x > world_w - margin: x = world_w - margin; angle = math.pi - angle + 0.6 * np.random.random() - 0.3
        if y < margin: y = margin; angle = -angle + 0.6 * np.random.random() - 0.3
        elif y > world_h - margin: y = world_h - margin; angle = -angle + 0.6 * np.random.random() - 0.3

        # Obstacle collision
        gx_new = int(x / cell); gy_new = int(y / cell)
        hit_obs = False
        if 0 <= gx_new < gw and 0 <= gy_new < gh:
            if obs_grid[gy_new, gx_new]:
                x = old_x; y = old_y; hit_obs = True
                angle += math.pi * 0.7 + (2.0 * np.random.random() - 1.0) * 0.8

        # Deposit pheromone (Disabled for MDP, Q-Learn; Modified for Lévy)
        gx = int(x / cell); gy = int(y / cell)
        if 0 <= gx < gw and 0 <= gy < gh and not obs_grid[gy, gx]:
            if algo_id in (ALGO_ACO, ALGO_MARKOV, ALGO_GENETIC):
                if has_food:
                    val = food_pher[gy, gx] + deposit
                    if val > max_pher: val = max_pher
                    food_pher[gy, gx] = val
                else:
                    val = home_pher[gy, gx] + deposit
                    if val > max_pher: val = max_pher
                    home_pher[gy, gx] = val
            elif algo_id == ALGO_LEVY:
                if ant_levy_timer[i] > 10:
                    pher_val = 30.0
                    if has_food: food_pher[gy, gx] = max(food_pher[gy, gx], pher_val)
                    else: home_pher[gy, gx] = max(home_pher[gy, gx], pher_val)
        deposit *= 0.995

        # Q-Learning Reward Calculation & Update
        if algo_id == ALGO_QLEARN and 0 <= gx_cur < gw and 0 <= gy_cur < gh:
            reward = -0.1 # Step penalty
            if hit_obs: reward = -5.0
            
            picked_food = False
            if not has_food:
                for j in range(n_foods):
                    if food_amount[j] > 0:
                        dx = x - food_x[j]; dy = y - food_y[j]
                        if dx * dx + dy * dy < food_r_sq: reward = 100.0; picked_food = True; break
            else:
                dx = x - nest_x; dy = y - nest_y
                if dx * dx + dy * dy < nest_r_sq: reward = 100.0

            if 0 <= gx_new < gw and 0 <= gy_new < gh:
                max_q_new = max(Q_table[gy_new, gx_new, 0], Q_table[gy_new, gx_new, 1], Q_table[gy_new, gx_new, 2])
                Q_table[gy_cur, gx_cur, action] += 0.5 * (reward + 0.9 * max_q_new - Q_table[gy_cur, gx_cur, action])

        # Pick up food
        if not has_food:
            for j in range(n_foods):
                if food_amount[j] > 0:
                    dx = x - food_x[j]; dy = y - food_y[j]
                    if dx * dx + dy * dy < food_r_sq:
                        has_food = True; food_amount[j] -= 1
                        angle += math.pi; deposit = pher_deposit
                        if algo_id == ALGO_GENETIC: ant_fitness[i] += 1
                        break
        else:
            dx = x - nest_x; dy = y - nest_y
            if dx * dx + dy * dy < nest_r_sq:
                has_food = False; collected_count += 1
                angle += math.pi; deposit = pher_deposit
                if algo_id == ALGO_GENETIC: ant_fitness[i] += 2

        # Genetic Algorithm Reproduction
        if algo_id == ALGO_GENETIC and ant_fitness[i] > 3:
            j = np.random.randint(0, n_active)
            if ant_fitness[j] < 1:
                ant_genes[j, 0] = max(0.1, min(ant_genes[i, 0] + (np.random.random() - 0.5) * 0.1, 0.8))
                ant_genes[j, 1] = max(0.3, min(ant_genes[i, 1] + (np.random.random() - 0.5) * 0.1, 1.2))
                ant_genes[j, 2] = max(0.02, min(ant_genes[i, 2] + (np.random.random() - 0.5) * 0.05, 0.3))
                ant_fitness[j] = 0 

        ant_x[i] = x; ant_y[i] = y; ant_angle[i] = angle
        ant_has_food[i] = has_food; ant_deposit[i] = deposit

    return collected_count


@njit(cache=True)
def update_pheromones(food_pher, home_pher, food_pher_tmp, home_pher_tmp,
                      evap_rate, diff_k, nmask, nvals, obs_grid):
    H, W = food_pher.shape
    for y in range(H):
        for x in range(W):
            if obs_grid[y, x]: food_pher[y, x] = 0.0; home_pher[y, x] = 0.0; continue
            fp = food_pher[y, x] * evap_rate; hp = home_pher[y, x] * evap_rate
            if fp < 0.05: fp = 0.0
            if hp < 0.05: hp = 0.0
            food_pher[y, x] = fp; home_pher[y, x] = hp

    for y in range(H):
        for x in range(W):
            if obs_grid[y, x]: food_pher_tmp[y, x] = 0.0; home_pher_tmp[y, x] = 0.0; continue
            fp_tot = 0.0; hp_tot = 0.0; cnt = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0: continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not obs_grid[ny, nx]:
                        fp_tot += food_pher[ny, nx]; hp_tot += home_pher[ny, nx]; cnt += 1
            if cnt > 0:
                food_pher_tmp[y, x] = food_pher[y, x] + (fp_tot / cnt - food_pher[y, x]) * diff_k
                home_pher_tmp[y, x] = home_pher[y, x] + (hp_tot / cnt - home_pher[y, x]) * diff_k
            else:
                food_pher_tmp[y, x] = food_pher[y, x]; home_pher_tmp[y, x] = home_pher[y, x]
            if nmask[y, x] and home_pher_tmp[y, x] < nvals[y, x]:
                home_pher_tmp[y, x] = nvals[y, x]


@njit(cache=True)
def mdp_value_iter(V, target_mask, obs_grid, gamma, n_iter):
    H, W = V.shape
    for _ in range(n_iter):
        for y in range(H):
            for x in range(W):
                if obs_grid[y, x] or target_mask[y, x]: continue
                max_val = -1e9
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dx == 0 and dy == 0: continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not obs_grid[ny, nx]:
                            dist = math.sqrt(dx*dx + dy*dy)
                            val = -0.1 * dist + gamma * V[ny, nx]
                            if val > max_val: max_val = val
                if max_val > -1e8: V[y, x] = max_val


@njit(cache=True)
def decay_grid(grid, rate):
    H = grid.shape[0]
    W = grid.shape[1]
    for y in range(H):
        for x in range(W):
            grid[y, x] *= rate


# ══════════════════════════════════════════════════════════════
# SIMULATION WIDGET
# ══════════════════════════════════════════════════════════════

class SimulationWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setFixedSize(WORLD_W, WORLD_H)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self.paused            = False
        self.show_pheromones   = True
        self.detailed_ants     = True
        self.auto_spawn        = False
        self.sim_speed         = 1
        self.pheromone_scheme  = "Classic"
        self.overlay_mode      = "Pheromones"
        self.algorithm         = "ACO (Standard)"
        self.mdp_gamma         = 0.95

        self.ant_speed_val     = 2.0
        self.evap_rate_val    = 0.996
        self.pher_deposit_val = 120.0
        self.diffuse_val      = 0.04
        self.turn_rate_val    = 0.4
        self.wander_noise_val = 0.15

        yy, xx = np.mgrid[0:GH, 0:GW]
        cx = xx * CELL + CELL * 0.5; cy = yy * CELL + CELL * 0.5
        self._ndist = np.sqrt((cx - NEST_X) ** 2 + (cy - NEST_Y) ** 2)
        self._nmask = (self._ndist < NEST_PHER_RADIUS).astype(np.bool_)
        self._nvals = NEST_PHER_STRENGTH * (1.0 - self._ndist / NEST_PHER_RADIUS)
        self._nvals[~self._nmask] = 0.0

        self._fps_frames = 0; self._fps_time = time.time(); self._fps_value = 0.0
        self._rate_history = deque(maxlen=60); self._last_collected = 0; self._rate_timer = time.time()
        self._frame = 0; self._drawing_obs = False; self._last_auto_spawn = time.time()

        self._init_world()

        print("Compiling Numba kernels …")
        self._step_ants(); self._evaporate()
        self._update_mdp(self._build_target_mask())
        print("Compilation complete.")

        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(16)

    def _init_world(self):
        self.food_pher = np.zeros((GH, GW), dtype=np.float32)
        self.home_pher = self._nvals.astype(np.float32).copy()
        self.food_pher_tmp = np.zeros_like(self.food_pher); self.home_pher_tmp = np.zeros_like(self.home_pher)

        self.ant_x = np.empty(MAX_ANTS, dtype=np.float32); self.ant_y = np.empty(MAX_ANTS, dtype=np.float32)
        self.ant_angle = np.random.uniform(0, 2 * math.pi, MAX_ANTS).astype(np.float32)
        self.ant_has_food = np.zeros(MAX_ANTS, dtype=np.bool_)
        self.ant_deposit = np.full(MAX_ANTS, self.pher_deposit_val, dtype=np.float32)
        self.ant_levy_timer = np.zeros(MAX_ANTS, dtype=np.int32)
        
        # Genetic Algorithm Arrays
        self.ant_genes = np.random.uniform(0.2, 0.8, (MAX_ANTS, 3)).astype(np.float32)
        self.ant_genes[:, 1] = np.random.uniform(0.4, 1.0, MAX_ANTS) # Sensor Angle
        self.ant_genes[:, 2] = np.random.uniform(0.05, 0.3, MAX_ANTS) # Wander Noise
        self.ant_fitness = np.zeros(MAX_ANTS, dtype=np.int32)

        r = np.random.uniform(0, NEST_R * 0.7, MAX_ANTS).astype(np.float32)
        a = np.random.uniform(0, 2 * math.pi, MAX_ANTS).astype(np.float32)
        self.ant_x = NEST_X + np.cos(a) * r; self.ant_y = NEST_Y + np.sin(a) * r
        self.n_active_ants = DEFAULT_ANTS

        self.food_x = np.zeros(MAX_FOODS, dtype=np.float32); self.food_y = np.zeros(MAX_FOODS, dtype=np.float32)
        self.food_amount = np.zeros(MAX_FOODS, dtype=np.int32); self.food_max_amount = np.zeros(MAX_FOODS, dtype=np.int32)
        self.n_foods = 0
        for fx, fy in [(140, 120), (820, 120), (140, 520), (820, 520), (480, 80), (480, 560)]:
            self._add_food(fx, fy)

        self.obs_grid = np.zeros((GH, GW), dtype=np.bool_); self.obs_cx = []; self.obs_cy = []

        self.V_food = np.zeros((GH, GW), dtype=np.float32); self.V_nest = np.zeros((GH, GW), dtype=np.float32)
        self.Q_table = np.zeros((GH, GW, 3), dtype=np.float32)
        self.visit_grid = np.zeros((GH, GW), dtype=np.float32)

        self.collected = 0; self._last_collected = 0; self._rate_history.clear(); self._rate_timer = time.time()

    def _add_food(self, x, y, amount=FOOD_AMOUNT):
        for i in range(self.n_foods):
            if self.food_amount[i] == 0:
                self.food_x[i] = x; self.food_y[i] = y; self.food_amount[i] = amount; self.food_max_amount[i] = amount; return
        if self.n_foods < MAX_FOODS:
            idx = self.n_foods; self.food_x[idx] = x; self.food_y[idx] = y
            self.food_amount[idx] = amount; self.food_max_amount[idx] = amount; self.n_foods += 1

    def _add_obstacle(self, cx, cy):
        if (cx - NEST_X) ** 2 + (cy - NEST_Y) ** 2 < (NEST_R + OBS_RADIUS + 10) ** 2: return
        self.obs_cx.append(float(cx)); self.obs_cy.append(float(cy))
        gx_c = int(cx / CELL); gy_c = int(cy / CELL); gr = int(OBS_RADIUS / CELL) + 2
        for dy in range(-gr, gr + 1):
            for dx in range(-gr, gr + 1):
                gx = gx_c + dx; gy = gy_c + dy
                if 0 <= gx < GW and 0 <= gy < GH:
                    px = gx * CELL + CELL * 0.5; py = gy * CELL + CELL * 0.5
                    if (px - cx) ** 2 + (py - cy) ** 2 < OBS_RADIUS ** 2: self.obs_grid[gy, gx] = True

    def _clear_obstacles(self): self.obs_cx.clear(); self.obs_cy.clear(); self.obs_grid[:] = False

    def set_ant_count(self, count):
        count = max(10, min(count, MAX_ANTS))
        if count > self.n_active_ants:
            start = self.n_active_ants; n_new = count - start
            r = np.random.uniform(0, NEST_R * 0.7, n_new).astype(np.float32)
            a = np.random.uniform(0, 2 * math.pi, n_new).astype(np.float32)
            self.ant_x[start:count] = NEST_X + np.cos(a) * r; self.ant_y[start:count] = NEST_Y + np.sin(a) * r
            self.ant_angle[start:count] = np.random.uniform(0, 2 * math.pi, n_new).astype(np.float32)
            self.ant_has_food[start:count] = False; self.ant_deposit[start:count] = self.pher_deposit_val
        self.n_active_ants = count

    def reset(self): self._init_world()

    def _tick(self):
        if self.paused: self.update(); return
        for _ in range(self.sim_speed):
            self._step_ants(); self._evaporate()
            if self.algorithm == "MDP Value Iteration": self._update_mdp(self._build_target_mask())
            decay_grid(self.visit_grid, 0.995)
            if self.algorithm == "Q-Learning": decay_grid(self.Q_table, 0.9999)

        self._frame += 1; self._fps_frames += 1; now = time.time()
        if now - self._fps_time >= 1.0: self._fps_value = self._fps_frames / (now - self._fps_time); self._fps_frames = 0; self._fps_time = now
        if now - self._rate_timer >= 1.0:
            rate = self.collected - self._last_collected; self._rate_history.append(rate); self._last_collected = self.collected; self._rate_timer = now
        if self.auto_spawn: self._maybe_auto_spawn(now)
        self.update()

    def _build_target_mask(self):
        mask = np.zeros((GH, GW), dtype=np.bool_); mask[int(NEST_Y/CELL), int(NEST_X/CELL)] = True
        for i in range(self.n_foods):
            if self.food_amount[i] > 0:
                gx, gy = int(self.food_x[i] / CELL), int(self.food_y[i] / CELL)
                if 0 <= gx < GW and 0 <= gy < GH: mask[gy, gx] = True
        return mask

    def _maybe_auto_spawn(self, now):
        if now - self._last_auto_spawn < 8.0: return
        active = int(np.count_nonzero(self.food_amount[:self.n_foods] > 0))
        if active >= 4: return
        for _ in range(100):
            x = np.random.uniform(60, WORLD_W - 60); y = np.random.uniform(60, WORLD_H - 60)
            gx, gy = int(x / CELL), int(y / CELL)
            if 0 <= gx < GW and 0 <= gy < GH and not self.obs_grid[gy, gx]:
                if (x - NEST_X) ** 2 + (y - NEST_Y) ** 2 > (NEST_R * 4) ** 2:
                    self._add_food(float(x), float(y)); self._last_auto_spawn = now; return

    def _step_ants(self):
        algo_map = {"ACO (Standard)": ALGO_ACO, "Markov Chain": ALGO_MARKOV, "MDP Value Iteration": ALGO_MDP,
                    "Q-Learning": ALGO_QLEARN, "Lévy Flight": ALGO_LEVY, "Genetic Algorithm": ALGO_GENETIC}
        algo_id = algo_map.get(self.algorithm, ALGO_ACO)
        self.collected = step_ants(
            self.ant_x, self.ant_y, self.ant_angle, self.ant_has_food, self.ant_deposit, self.ant_levy_timer,
            self.ant_genes, self.ant_fitness,
            self.food_pher, self.home_pher, self.food_x, self.food_y, self.food_amount,
            self.n_foods, self.collected, self.obs_grid, self.visit_grid,
            self.V_food, self.V_nest, self.Q_table,
            NEST_X, NEST_Y, NEST_R ** 2, FOOD_R ** 2,
            WORLD_W, WORLD_H, CELL, GW, GH,
            SENSOR_DIST, self.turn_rate_val, self.wander_noise_val,
            self.ant_speed_val, MAX_PHER, self.pher_deposit_val, self.n_active_ants, algo_id)

    def _evaporate(self):
        update_pheromones(self.food_pher, self.home_pher, self.food_pher_tmp, self.home_pher_tmp,
                          self.evap_rate_val, self.diffuse_val, self._nmask, self._nvals, self.obs_grid)
        self.food_pher, self.food_pher_tmp = self.food_pher_tmp, self.food_pher
        self.home_pher, self.home_pher_tmp = self.home_pher_tmp, self.home_pher

    def _update_mdp(self, target_mask):
        food_mask = np.zeros((GH, GW), dtype=np.bool_)
        for i in range(self.n_foods):
            if self.food_amount[i] > 0:
                gx, gy = int(self.food_x[i]/CELL), int(self.food_y[i]/CELL)
                if 0 <= gx < GW and 0 <= gy < GH: food_mask[gy, gx] = True
        self.V_food[food_mask] = 100.0; mdp_value_iter(self.V_food, food_mask, self.obs_grid, self.mdp_gamma, n_iter=20)
        nest_mask = np.zeros((GH, GW), dtype=np.bool_); nest_mask[int(NEST_Y/CELL), int(NEST_X/CELL)] = True
        self.V_nest[nest_mask] = 100.0; mdp_value_iter(self.V_nest, nest_mask, self.obs_grid, self.mdp_gamma, n_iter=20)

    # ── rendering ────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.fillRect(self.rect(), QColor(18, 14, 10))
        self._render_overlays(p); self._render_obstacles(p); self._render_nest(p); self._render_food(p); self._render_ants(p)
        self._render_graph(p); self._render_hud(p)
        if self.paused:
            p.fillRect(self.rect(), QColor(0, 0, 0, 80)); p.setPen(QColor(255, 255, 255, 220))
            p.setFont(QFont("Sans", 36, QFont.Bold)); p.drawText(self.rect(), Qt.AlignCenter, "PAUSED")
        p.end()

    def _render_overlays(self, p):
        if self.overlay_mode == "Pheromones" and self.show_pheromones: self._render_pheromones(p)
        elif self.overlay_mode == "MDP Values": self._render_mdp_values(p)
        elif self.overlay_mode == "Stationary Dist": self._render_stationary_dist(p)
        elif self.overlay_mode == "Q-Table": self._render_q_table(p)

    def _render_pheromones(self, p):
        fp = np.clip(self.food_pher, 0, 300); hp = np.clip(self.home_pher, 0, 300)
        scheme = self.pheromone_scheme
        if scheme == "Thermal": r = np.clip(fp*1.2,0,255); g = np.clip(fp*0.55+hp*0.25,0,200); b = np.clip(hp*1.0,0,240); a = np.clip(np.maximum(fp,hp)*0.6,0,180)
        elif scheme == "Neon": r = np.clip(hp*0.9,0,230); g = np.clip(fp*0.9+hp*0.05,0,240); b = np.clip(fp*0.9,0,230); a = np.clip(np.maximum(fp,hp)*0.65,0,190)
        else: r = np.clip(fp*0.85+hp*0.05,0,230); g = np.clip(fp*0.35,0,100); b = np.clip(hp*0.70,0,210); a = np.clip(np.maximum(fp,hp)*0.55,0,160)
        self._draw_grid_rgba(p, r, g, b, a)

    def _render_mdp_values(self, p):
        v = np.clip(self.V_food + self.V_nest, -50, 100); norm = np.clip((v + 50) / 150.0, 0, 1)
        r = np.clip(norm*255*0.2,0,255); g = np.clip(norm*255*0.8,0,255); b = np.clip(norm*255*1.0,0,255); a = np.clip(norm*200,0,200)
        self._draw_grid_rgba(p, r, g, b, a)

    def _render_stationary_dist(self, p):
        v = np.clip(self.visit_grid, 0, 500); norm = v / 500.0
        r = np.clip(norm*255,0,255); g = np.clip(norm*100,0,255); b = np.clip(norm*50,0,255); a = np.clip(norm*220,0,220)
        self._draw_grid_rgba(p, r, g, b, a)

    def _render_q_table(self, p):
        v = np.max(self.Q_table, axis=2); v = np.clip(v, -10, 100); norm = np.clip((v + 10) / 110.0, 0, 1)
        r = np.clip(norm*255*1.0,0,255); g = np.clip(norm*255*0.2,0,255); b = np.clip(norm*255*0.8,0,255); a = np.clip(norm*200,0,200)
        self._draw_grid_rgba(p, r, g, b, a)

    def _draw_grid_rgba(self, p, r, g, b, a):
        buf = np.empty((GH, GW, 4), dtype=np.uint8)
        buf[:, :, 0] = r.astype(np.uint8); buf[:, :, 1] = g.astype(np.uint8)
        buf[:, :, 2] = b.astype(np.uint8); buf[:, :, 3] = a.astype(np.uint8)
        buf = np.ascontiguousarray(buf); img = QImage(buf.data, GW, GH, GW * 4, QImage.Format_RGBA8888)
        self._buf = buf; p.drawImage(0, 0, img.scaled(WORLD_W, WORLD_H, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

    def _render_nest(self, p):
        pulse = 1.0 + 0.04 * math.sin(self._frame * 0.06); nr = NEST_R * pulse
        p.setPen(Qt.NoPen); p.setBrush(QColor(200, 140, 50, 25)); p.drawEllipse(QPointF(NEST_X, NEST_Y), nr * 1.6, nr * 1.6)
        grad = QRadialGradient(QPointF(NEST_X, NEST_Y), nr)
        grad.setColorAt(0.0, QColor(200, 140, 60)); grad.setColorAt(0.5, QColor(150, 95, 40)); grad.setColorAt(1.0, QColor(80, 50, 15, 120))
        p.setBrush(QBrush(grad)); p.drawEllipse(QPointF(NEST_X, NEST_Y), nr, nr)
        p.setBrush(QColor(35, 22, 8)); p.drawEllipse(QPointF(NEST_X, NEST_Y), nr * 0.35, nr * 0.35)

    def _render_food(self, p):
        t = self._frame * 0.03
        for i in range(self.n_foods):
            amt = self.food_amount[i]
            if amt <= 0: continue
            x, y, max_amt = float(self.food_x[i]), float(self.food_y[i]), self.food_max_amount[i]
            ratio = amt / max(1, max_amt); pulse = 1.0 + 0.06 * math.sin(t + i * 1.3); sz = FOOD_R * (0.3 + 0.7 * ratio) * pulse
            p.setPen(Qt.NoPen); p.setBrush(QColor(50, 200, 50, 35)); p.drawEllipse(QPointF(x, y), sz * 2.0, sz * 2.0)
            grad = QRadialGradient(QPointF(x, y), sz)
            if ratio < 0.25: grad.setColorAt(0.0, QColor(240, 200, 60)); grad.setColorAt(0.6, QColor(200, 150, 30)); grad.setColorAt(1.0, QColor(140, 100, 10, 80)); p.setPen(QPen(QColor(200, 160, 30, 150), 0.8))
            else: grad.setColorAt(0.0, QColor(100, 240, 100)); grad.setColorAt(0.6, QColor(50, 190, 50)); grad.setColorAt(1.0, QColor(20, 110, 20, 80)); p.setPen(QPen(QColor(40, 160, 40, 150), 0.8))
            p.setBrush(QBrush(grad)); p.drawEllipse(QPointF(x, y), sz, sz)
            p.setPen(QColor(230, 255, 230, 200)); p.setFont(QFont("Sans", 8)); p.drawText(QRectF(x - 18, y - 6, 36, 12), Qt.AlignCenter, str(amt))

    def _render_obstacles(self, p):
        for cx, cy in zip(self.obs_cx, self.obs_cy):
            grad = QRadialGradient(QPointF(cx - 2, cy - 2), OBS_RADIUS)
            grad.setColorAt(0.0, QColor(105, 100, 95)); grad.setColorAt(0.5, QColor(72, 67, 62)); grad.setColorAt(1.0, QColor(42, 37, 32))
            p.setPen(QPen(QColor(55, 50, 45), 0.8)); p.setBrush(QBrush(grad)); p.drawEllipse(QPointF(cx, cy), OBS_RADIUS, OBS_RADIUS)

    def _render_ants(self, p):
        n = self.n_active_ants
        if self.detailed_ants and self.algorithm != "Genetic Algorithm": self._render_ants_detailed(p, n)
        else: self._render_ants_fast(p, n)

    def _render_ants_detailed(self, p, n):
        p.setPen(Qt.NoPen)
        for i in range(n):
            x, y, angle, has = float(self.ant_x[i]), float(self.ant_y[i]), float(self.ant_angle[i]), bool(self.ant_has_food[i])
            p.save(); p.translate(x, y); p.rotate(math.degrees(angle))
            if has: p.setBrush(QColor(210, 120, 40)); p.drawEllipse(QPointF(-1, 0), 2.5, 1.5); p.setBrush(QColor(180, 100, 35)); p.drawEllipse(QPointF(1.5, 0), 1.2, 1.0); p.drawEllipse(QPointF(3, 0), 1.0, 0.9); p.setBrush(QColor(80, 210, 80)); p.drawEllipse(QPointF(4, 0), 0.8, 0.8)
            else: p.setBrush(QColor(130, 110, 85)); p.drawEllipse(QPointF(-1, 0), 2.5, 1.5); p.setBrush(QColor(150, 130, 100)); p.drawEllipse(QPointF(1.5, 0), 1.2, 1.0); p.drawEllipse(QPointF(3, 0), 1.0, 0.9)
            p.restore()

    def _render_ants_fast(self, p, n):
        p.setPen(Qt.NoPen)
        if self.algorithm == "Genetic Algorithm":
            max_fit = max(1, int(np.max(self.ant_fitness[:n])))
            for i in range(n):
                fit = self.ant_fitness[i]; ratio = fit / max_fit
                r_c = int(255 * (1 - ratio)); g_c = int(255 * ratio * 0.8 + 50); b_c = int(255 * ratio)
                p.setBrush(QColor(r_c, g_c, b_c)); p.drawEllipse(QPointF(float(self.ant_x[i]), float(self.ant_y[i])), 2.2, 1.5)
        else:
            p.setBrush(QColor(140, 120, 90))
            for i in range(n):
                if not self.ant_has_food[i]: p.drawEllipse(QPointF(float(self.ant_x[i]), float(self.ant_y[i])), 2.0, 1.4)
            p.setBrush(QColor(220, 130, 40))
            for i in range(n):
                if self.ant_has_food[i]: p.drawEllipse(QPointF(float(self.ant_x[i]), float(self.ant_y[i])), 2.2, 1.5)

    def _render_graph(self, p):
        data = list(self._rate_history)
        if len(data) < 2: return
        x0, y0, w, h = WORLD_W - 185, WORLD_H - 65, 175, 50
        p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 150)); p.drawRoundedRect(QRectF(x0 - 6, y0 - 16, w + 12, h + 22), 5, 5)
        p.setPen(QColor(180, 180, 180, 200)); p.setFont(QFont("Sans", 7)); p.drawText(QRectF(x0, y0 - 14, w, 12), Qt.AlignLeft, "Collection Rate / sec")
        max_val = max(max(data), 1); pts = []
        for i, v in enumerate(data): px = x0 + i * w / (len(data) - 1); py = y0 + h - (v / max_val) * h * 0.9; pts.append(QPointF(px, py))
        p.setPen(Qt.NoPen); p.setBrush(QColor(60, 180, 60, 40)); p.drawPolygon(QPolygonF([QPointF(x0, y0 + h)] + pts + [QPointF(x0 + w, y0 + h)]))
        p.setPen(QPen(QColor(80, 220, 80, 200), 1.5)); p.setBrush(Qt.NoBrush)
        for j in range(1, len(pts)): p.drawLine(pts[j - 1], pts[j])

    def _render_hud(self, p):
        p.setFont(QFont("Sans", 10)); carrying = int(np.count_nonzero(self.ant_has_food[:self.n_active_ants]))
        active_foods = int(np.count_nonzero(self.food_amount[:self.n_foods] > 0)); total_remaining = int(np.sum(self.food_amount[:self.n_foods]))
        avg_rate = np.mean(list(self._rate_history)) if self._rate_history else 0
        lines = [
            f"Algorithm: {self.algorithm}   |   Overlay: {self.overlay_mode}",
            f"Collected: {self.collected}   Remaining: {total_remaining}   Rate: {avg_rate:.1f}/s",
            f"Ants: {self.n_active_ants}  (carrying: {carrying})   Sources: {active_foods}   FPS: {self._fps_value:.0f}",
        ]
        y = 20; p.setPen(QColor(220, 220, 220, 210))
        for line in lines: p.drawText(10, y, line); y += 18
        p.setPen(QColor(160, 160, 160, 150)); p.setFont(QFont("Sans", 8))
        p.drawText(10, WORLD_H - 10, "L-click: food  |  R-click: obstacle  |  Space: pause  |  R: reset  |  P: pheromones  |  C: clear obs")

    def mousePressEvent(self, event):
        pos = event.position()
        if event.button() == Qt.LeftButton: self._add_food(pos.x(), pos.y())
        elif event.button() == Qt.RightButton: self._drawing_obs = True; self._add_obstacle(pos.x(), pos.y())

    def mouseMoveEvent(self, event):
        if self._drawing_obs: self._add_obstacle(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton: self._drawing_obs = False

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space: self.paused = not self.paused
        elif key == Qt.Key_R: self.reset()
        elif key == Qt.Key_C: self._clear_obstacles()
        else: super().keyPressEvent(event)


# ══════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ant Colony Simulation (6 Algorithms)")

        self.setStyleSheet("""
            QMainWindow { background: #1a1a2e; }
            QPushButton { background: #16213e; color: #ddd; border: 1px solid #0f3460; padding: 6px 12px; border-radius: 4px; font-size: 12px; }
            QPushButton:hover { background: #0f3460; } QPushButton:pressed { background: #533483; } QPushButton:checked { background: #533483; border-color: #e94560; }
            QLabel { color: #bbb; font-size: 11px; }
            QComboBox { background: #16213e; color: #ddd; border: 1px solid #0f3460; padding: 4px 8px; border-radius: 3px; font-size: 11px; }
            QComboBox::drop-down { border: none; } QComboBox QAbstractItemView { background: #16213e; color: #ddd; selection-background-color: #0f3460; }
            QSlider::groove:horizontal { height: 5px; background: #16213e; border-radius: 2px; }
            QSlider::handle:horizontal { background: #e94560; width: 12px; margin: -4px 0; border-radius: 6px; }
        """)

        central = QWidget(); vbox = QVBoxLayout(); vbox.setContentsMargins(10, 10, 10, 10); vbox.setSpacing(4)
        self.sim = SimulationWidget(); vbox.addWidget(self.sim, alignment=Qt.AlignCenter)

        # ── Button row ──
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self.pause_btn = QPushButton("⏸ Pause"); self.pause_btn.setCheckable(True); self.pause_btn.clicked.connect(self._toggle_pause); btn_row.addWidget(self.pause_btn)
        reset_btn = QPushButton("🔄 Reset"); reset_btn.clicked.connect(self.sim.reset); btn_row.addWidget(reset_btn)
        clear_obs_btn = QPushButton("🧱 Clear Obstacles"); clear_obs_btn.clicked.connect(self.sim._clear_obstacles); btn_row.addWidget(clear_obs_btn)
        self.auto_btn = QPushButton("🍞 Auto-Food: OFF"); self.auto_btn.setCheckable(True); self.auto_btn.clicked.connect(self._toggle_auto); btn_row.addWidget(self.auto_btn)
        self.detail_btn = QPushButton("🐜 Detailed: ON"); self.detail_btn.setCheckable(True); self.detail_btn.setChecked(True); self.detail_btn.clicked.connect(self._toggle_detail); btn_row.addWidget(self.detail_btn)

        btn_row.addSpacing(10)
        btn_row.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["ACO (Standard)", "Markov Chain", "MDP Value Iteration", "Q-Learning", "Lévy Flight", "Genetic Algorithm"])
        self.algo_combo.currentTextChanged.connect(lambda s: setattr(self.sim, 'algorithm', s)); btn_row.addWidget(self.algo_combo)

        btn_row.addSpacing(5)
        btn_row.addWidget(QLabel("Overlay:"))
        self.overlay_combo = QComboBox(); self.overlay_combo.addItems(["Pheromones", "MDP Values", "Stationary Dist", "Q-Table"])
        self.overlay_combo.currentTextChanged.connect(lambda s: setattr(self.sim, 'overlay_mode', s)); btn_row.addWidget(self.overlay_combo)

        btn_row.addSpacing(5)
        btn_row.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox(); self.color_combo.addItems(["Classic", "Thermal", "Neon"])
        self.color_combo.currentTextChanged.connect(lambda s: setattr(self.sim, 'pheromone_scheme', s)); btn_row.addWidget(self.color_combo)

        btn_row.addStretch(); vbox.addLayout(btn_row)

        # ── Slider row ──
        slider_row = QHBoxLayout(); slider_row.setSpacing(8); self._sliders = {}
        slider_row.addLayout(self._make_slider("Speed", 1, 10, 1, lambda v: setattr(self.sim, 'sim_speed', v), fmt="{:.0f}", key="speed"))
        slider_row.addLayout(self._make_slider("Ants", 10, MAX_ANTS, DEFAULT_ANTS, self.sim.set_ant_count, fmt="{:.0f}", key="ants"))
        slider_row.addLayout(self._make_slider("Ant Spd", 5, 40, 20, lambda v: setattr(self.sim, 'ant_speed_val', v / 10.0), fmt="{:.1f}", key="ant_spd"))
        slider_row.addLayout(self._make_slider("Turn", 1, 10, 4, lambda v: setattr(self.sim, 'turn_rate_val', v / 10.0), fmt="{:.1f}", key="turn"))
        slider_row.addLayout(self._make_slider("Wander", 1, 30, 15, lambda v: setattr(self.sim, 'wander_noise_val', v / 100.0), fmt="{:.2f}", key="wander"))
        slider_row.addLayout(self._make_slider("Evap", 990, 999, 996, lambda v: setattr(self.sim, 'evap_rate_val', v / 1000.0), fmt="{:.3f}", key="evap"))
        slider_row.addLayout(self._make_slider("Diffuse", 1, 15, 4, lambda v: setattr(self.sim, 'diffuse_val', v / 100.0), fmt="{:.3f}", key="diffuse"))
        slider_row.addLayout(self._make_slider("MDP γ", 50, 99, 95, lambda v: setattr(self.sim, 'mdp_gamma', v / 100.0), fmt="{:.2f}", key="gamma"))

        slider_row.addStretch(); vbox.addLayout(slider_row)

        central.setLayout(vbox); self.setCentralWidget(central); self.adjustSize()

    def _make_slider(self, label, lo, hi, default, callback, fmt="{:.0f}", key=""):
        layout = QVBoxLayout(); layout.setSpacing(0); layout.setContentsMargins(2, 2, 2, 2)
        lbl = QLabel(f"{label}: {fmt.format(default)}"); lbl.setAlignment(Qt.AlignCenter); layout.addWidget(lbl)
        slider = QSlider(Qt.Horizontal); slider.setRange(lo, hi); slider.setValue(default); slider.setFixedWidth(90)
        def on_change(v): lbl.setText(f"{label}: {fmt.format(v)}"); callback(v)
        slider.valueChanged.connect(on_change); layout.addWidget(slider)
        self._sliders[key] = (slider, lbl); return layout

    def _toggle_pause(self):
        self.sim.paused = not self.sim.paused; self.pause_btn.setChecked(self.sim.paused)
        self.pause_btn.setText("▶ Resume" if self.sim.paused else "⏸ Pause")

    def _toggle_auto(self):
        self.sim.auto_spawn = not self.sim.auto_spawn; on = self.sim.auto_spawn; self.auto_btn.setChecked(on)
        self.auto_btn.setText(f"🍞 Auto-Food: {'ON' if on else 'OFF'}")
        if on: self.sim._last_auto_spawn = time.time()

    def _toggle_detail(self):
        self.sim.detailed_ants = not self.sim.detailed_ants; on = self.sim.detailed_ants; self.detail_btn.setChecked(on)
        self.detail_btn.setText(f"🐜 Detailed: {'ON' if on else 'OFF'}")

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space: self._toggle_pause()
        elif key == Qt.Key_R: self.sim.reset()
        elif key == Qt.Key_P: self.sim.show_pheromones = not self.sim.show_pheromones
        elif key == Qt.Key_C: self.sim._clear_obstacles()
        else: super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())