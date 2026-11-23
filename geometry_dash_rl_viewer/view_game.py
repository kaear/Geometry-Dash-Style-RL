"""Play or watch the exact starter GeometryDashEnv with a Pygame window.

Run from this directory:
    python view_game.py --mode manual
    python view_game.py --mode heuristic
    python view_game.py --mode ppo --model models/ppo_fixed_level.zip
"""
import argparse
import math
from pathlib import Path

import pygame
from geometry_env import GeometryDashEnv

WIDTH, HEIGHT = 1000, 410
GROUND_Y = 310
SCALE = 35.0  # pixels per environment unit
PLAYER_SIZE = 19
BG = (17, 25, 39)
GROUND = (38, 60, 82)
EDGE = (85, 127, 158)
CYAN = (72, 206, 232)
SPIKE = (255, 149, 101)
WHITE = (230, 238, 247)
MUTED = (151, 167, 185)


def choose_action(mode, env, observation, model, keys):
    if mode == 'manual':
        return 1 if keys[pygame.K_SPACE] or keys[pygame.K_UP] else 0
    if mode == 'random':
        return int(env.action_space.sample())
    if mode == 'heuristic':
        # The same simple jump-timing heuristic used in smoke_test.py.
        next_dist = float(observation[3]) * 25.0
        grounded = bool(observation[2] > 0.5)
        return int(grounded and 1.5 < next_dist < 2.7)
    action, _ = model.predict(observation, deterministic=True)
    return int(action)


def main():
    parser = argparse.ArgumentParser(description='Geometry Dash RL live viewer')
    parser.add_argument('--mode', choices=['manual', 'random', 'heuristic', 'ppo'], default='manual')
    parser.add_argument('--model', default='models/ppo_fixed_level.zip')
    parser.add_argument('--fps', type=int, default=45, help='Simulation steps per second, 1-240')
    parser.add_argument('--reward-mode', choices=['sparse', 'shaped'], default='sparse')
    args = parser.parse_args()
    if not (1 <= args.fps <= 240):
        parser.error('--fps must be between 1 and 240')

    model = None
    if args.mode == 'ppo':
        if not Path(args.model).exists():
            parser.error(f'PPO checkpoint not found: {args.model}  (run python train_ppo.py first)')
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device='cpu')

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('CS 175 Geometry Dash — '+args.mode.upper())
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('arial', 20)
    small = pygame.font.SysFont('arial', 15)
    env = GeometryDashEnv(reward_mode=args.reward_mode)
    obs, info = env.reset(seed=0)
    wins, attempts, steps = 0, 1, 0
    paused, finished = False, False
    last_action = 0
    outcome = ''

    def restart():
        nonlocal obs, info, attempts, steps, finished, outcome, paused
        obs, info = env.reset(seed=attempts)
        attempts += 1
        steps = 0
        finished = False
        outcome = ''
        paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    restart()
                elif event.key == pygame.K_p:
                    paused = not paused

        if not paused and not finished:
            last_action = choose_action(args.mode, env, obs, model, pygame.key.get_pressed())
            obs, reward, terminated, truncated, info = env.step(last_action)
            steps += 1
            if terminated or truncated:
                finished = True
                if info.get('success'):
                    wins += 1
                    outcome = 'LEVEL COMPLETE!'
                elif info.get('collision'):
                    outcome = 'COLLISION — press R to try again'
                else:
                    outcome = 'TIME LIMIT — press R to try again'

        screen.fill(BG)
        # Camera follows player, beginning after the first few world units.
        cam_x = max(0.0, env.x * SCALE - WIDTH * 0.25)
        pygame.draw.rect(screen, GROUND, (0, GROUND_Y, WIDTH, HEIGHT-GROUND_Y))
        pygame.draw.line(screen, EDGE, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)
        # Ground markers convey horizontal movement visually.
        for t in range(0, 32):
            px = int(t * SCALE - cam_x)
            if -20 <= px <= WIDTH:
                pygame.draw.line(screen, (53, 83, 109), (px, GROUND_Y+8), (px, HEIGHT), 1)

        for pos, width, height in env.obstacles:
            px = int(pos * SCALE - cam_x)
            obstacle_w = max(8, int(width * SCALE))
            obstacle_h = max(9, int(height * SCALE))
            pygame.draw.polygon(screen, SPIKE,
                                [(px, GROUND_Y), (px+obstacle_w//2, GROUND_Y-obstacle_h),
                                 (px+obstacle_w, GROUND_Y)])
        finish_x = int(env.level_end * SCALE - cam_x)
        pygame.draw.line(screen, (133, 233, 151), (finish_x, 55), (finish_x, GROUND_Y), 3)
        end_text = small.render('FINISH', True, (133, 233, 151))
        screen.blit(end_text, (finish_x-20, 40))

        player_x = int(env.x*SCALE-cam_x)
        player_y = int(GROUND_Y-env.y*SCALE-PLAYER_SIZE)
        pygame.draw.rect(screen, CYAN, (player_x, player_y, PLAYER_SIZE, PLAYER_SIZE), border_radius=3)
        pygame.draw.rect(screen, WHITE, (player_x+4, player_y+4, 4, 4), border_radius=1)
        pygame.draw.rect(screen, WHITE, (player_x+11, player_y+4, 4, 4), border_radius=1)

        pygame.draw.rect(screen, (27, 40, 56), (0, 0, WIDTH, 92))
        ui = f'Mode: {args.mode.upper()}    Progress: {info.get("progress", 0):.1%}    Attempt: {attempts}    Wins: {wins}    Action: {"JUMP" if last_action else "WAIT"}'
        screen.blit(font.render(ui, True, WHITE), (18, 15))
        tip = 'SPACE/UP jump (manual)  |  P pause  |  R restart  |  ESC exit'
        screen.blit(small.render(tip, True, MUTED), (18, 49))
        if finished:
            text = font.render(outcome, True, (133, 233, 151) if info.get('success') else SPIKE)
            screen.blit(text, (18, 70))
        elif paused:
            screen.blit(font.render('PAUSED', True, SPIKE), (18, 70))
        pygame.display.flip()
        clock.tick(args.fps)

    env.close()
    pygame.quit()


if __name__ == '__main__':
    main()
