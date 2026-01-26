 1. Klassische Low‑Dimensional Environments (Vektorzustände)
CartPole, MountainCar, Acrobot, LunarLander
Diese Umgebungen haben kleine Zustandsräume und diskrete Aktionen. Hier geht es vor allem um Stabilität, Exploration und Overestimation‑Kontrolle.
CartPole-v1
- Beste Varianten:
- Double DQN
- Dueling DQN
- PER
- Warum:
- Overestimation ist ein Thema → DDQN
- Viele Aktionen sind ähnlich → Dueling
- PER beschleunigt das Lernen
- Overkill: Distributional, Noisy Nets, Rainbow (funktioniert, aber unnötig)

MountainCar-v0
- Beste Varianten:
- PER
- N‑Step DQN
- Double DQN
- Warum:
- Reward ist extrem spärlich → PER + N‑Step helfen massiv
- DDQN stabilisiert die Q‑Schätzung
- Optional: Noisy Nets für bessere Exploration

Acrobot-v1
- Beste Varianten:
- Double DQN
- Dueling DQN
- PER
- Warum:
- Hohe Dynamik, viele suboptimale Aktionen → Dueling
- PER beschleunigt das Finden guter Trajektorien

LunarLander-v2
- Beste Varianten:
- Double DQN
- Dueling DQN
- PER
- Noisy DQN
- Warum:
- Reward ist nicht spärlich, aber noisy → Noisy Nets helfen
- Dueling + DDQN sind fast Pflicht
- Sehr gut: Rainbow (hier lohnt es sich)

🔵 2. Pixel‑basierte Environments (Atari, MinAtar, Retro)
Breakout, Pong, Space Invaders, etc.
Hier brauchst du starke Feature‑Extraktion + stabile Q‑Schätzung.
Atari (ALE)
- Beste Varianten:
- Rainbow DQN (State of the Art)
- Distributional DQN (C51, QR‑DQN, IQN)
- Noisy Nets
- N‑Step
- Double + Dueling
- Warum:
- Pixelinput → CNN + Distributional RL ist extrem stark
- Exploration ist schwierig → Noisy Nets oder Bootstrapped DQN
- Multi‑Step verbessert Credit Assignment
Kurz: Für Atari ist Rainbow die Benchmark.

MinAtar
- Beste Varianten:
- Distributional DQN
- Noisy Nets
- PER
- Warum:
- Weniger komplex als Atari, aber gleiche Strukturen
- Distributional RL bringt hier besonders viel

🟠 3. Stochastische oder teilweise beobachtbare Umgebungen
Flickering Atari, POMDP‑Varianten, Env mit Masking
Flickering Atari / POMDP‑Varianten
- Beste Varianten:
- DRQN (Recurrent DQN)
- Bootstrapped DQN
- Noisy Nets
- Warum:
- LSTM/GRU kompensiert fehlende Beobachtungen
- Bootstrapped DQN liefert bessere Exploration bei Unsicherheit

🟣 4. Multi‑Agent‑Environments (z. B. PettingZoo mit diskreten Aktionen)
Kooperative Settings
- Beste Varianten:
- VDN / QMIX (Q‑Learning‑basiert, aber nicht klassisch DQN)
- Double DQN als Basis für einzelne Agenten
- Warum:
- Joint‑Action‑Spaces explodieren → Faktorisierung nötig

🟤 5. Environments mit sehr vielen diskreten Aktionen
z. B. Empfehlungssystem‑ähnliche Gym‑Envs
Parametric DQN
- Beste Varianten:
- Parametric DQN
- Dueling + PER
- Warum:
- Klassisches DQN skaliert schlecht bei 100+ Aktionen
- Parametric DQN modelliert Aktionen als Features

🟥 6. Nicht geeignet für DQN (aber oft gefragt)
Pendulum, Continuous Control, MuJoCo
→ Kein DQN‑Variant geeignet, da kontinuierliche Aktionen.
→ Nutze SAC, TD3, DDPG.

Empfohlene Pipeline je nach Komplexität

![alt text](gym_complex_pipeline.png)
