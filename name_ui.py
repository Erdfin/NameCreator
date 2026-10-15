import pygame
import csv
import sys
import random
import math
from pathlib import Path

from name_model import nameGeneratorModel

genModel = nameGeneratorModel("data.csv")
genModel.retrain_model()

# ----------------------------
# Configuration
# ----------------------------

DATA_PATH = Path("data.csv")

SCREEN_WIDTH = 720
SCREEN_HEIGHT = 420

BG_COLOR = (25, 25, 25)
TEXT_COLOR = (235, 235, 235)
BAR_BG = (60, 60, 60)

GREEN = (90, 210, 130)
RED = (230, 90, 90)
YELLOW = (230, 200, 90)
BLUE = (120, 160, 255)

FONT_BIG = 64
FONT_SMALL = 22

GEN_THRESHOLD = 0.8
UNCERTAINTY_PROB = 0.75  # probability of uncertainty sampling

# ----------------------------
# Dataset stats
# ----------------------------

accepted = 0
rejected = 0

if DATA_PATH.exists():
    with DATA_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "1":
                accepted += 1
            elif row["label"] == "0":
                rejected += 1

# ----------------------------
# Helpers
# ----------------------------

def append_to_dataset(word, label):
    global accepted, rejected

    exists = DATA_PATH.exists()
    with DATA_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["string", "label"])
        writer.writerow([word, label])

    if label == 1:
        accepted += 1
    else:
        rejected += 1


def entropy(p):
    """Binary entropy, max at p=0.5"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def get_candidate():
    while True:
        candidates = genModel.generate_names(n=50, threshold=GEN_THRESHOLD)
        scores = genModel.predict_batch(candidates)
        yield from zip(candidates, scores)


def confidence_color(score):
    if score >= 0.7:
        return GREEN
    elif score <= 0.3:
        return RED
    return YELLOW


# ----------------------------
# Pygame setup
# ----------------------------

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Name Labeling Tool (Active Learning)")

font_big = pygame.font.SysFont(None, FONT_BIG)
font_small = pygame.font.SysFont(None, FONT_SMALL)

clock = pygame.time.Clock()

canditate_gen = get_candidate()
canditate_gen.send(None)
current_name, current_score = canditate_gen.send(None)
print(current_name, current_score)
message = ""

# ----------------------------
# Main loop
# ----------------------------

running = True
while running:
    screen.fill(BG_COLOR)

    # Name
    name_surf = font_big.render(current_name, True, TEXT_COLOR)
    name_rect = name_surf.get_rect(center=(SCREEN_WIDTH // 2, 80))
    screen.blit(name_surf, name_rect)

    # Confidence bar
    bar_x, bar_y = 160, 150
    bar_w, bar_h = 400, 26

    pygame.draw.rect(screen, BAR_BG, (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(
        screen,
        confidence_color(current_score),
        (bar_x, bar_y, int(bar_w * current_score), bar_h),
    )

    score_text = font_small.render(
        f"Model confidence: {current_score:.3f}", True, TEXT_COLOR
    )
    screen.blit(score_text, (bar_x, bar_y - 28))

    # Stats
    total = accepted + rejected
    acc_rate = accepted / total if total else 0.0

    stats = (
        f"Accepted: {accepted}   "
        f"Rejected: {rejected}   "
        f"Acceptance rate: {acc_rate:.2%}"
    )
    stats_surf = font_small.render(stats, True, TEXT_COLOR)
    screen.blit(stats_surf, (40, 210))

    # Instructions
    instructions = [
        "(left) Reject",
        "(right) Accept",
        "(down) Skip",
        "(T) Retrain model",
        "(Q) Quit",
    ]

    for i, txt in enumerate(instructions):
        surf = font_small.render(txt, True, TEXT_COLOR)
        screen.blit(surf, (40 + i * 130, 270))

    # Feedback
    if message:
        msg_color = GREEN if "Accepted" in message else RED if "Rejected" in message else BLUE
        msg_surf = font_small.render(message, True, msg_color)
        screen.blit(msg_surf, (SCREEN_WIDTH - 260, 210))

    pygame.display.flip()

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

            elif event.key == pygame.K_RIGHT:
                append_to_dataset(current_name, 1)
                message = "Accepted ✓"
                current_name, current_score = canditate_gen.send(None)

            elif event.key == pygame.K_LEFT:
                append_to_dataset(current_name, 0)
                message = "Rejected ✗"
                current_name, current_score = canditate_gen.send(None)

            elif event.key == pygame.K_DOWN:
                message = "Skipped"
                current_name, current_score = canditate_gen.send(None)

            elif event.key == pygame.K_t:
                message = "Retraining model..."
                pygame.display.flip()
                genModel.retrain_model()
                message = "Model retrained ✓"
                current_name, current_score = canditate_gen.send(None)

    clock.tick(30)

pygame.quit()
sys.exit()
