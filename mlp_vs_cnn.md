🔷 MLP – Vollständig verbundene Schichten
Ein MLP besteht aus:
- Dense‑Layers (jede Neuron‑zu‑Neuron‑Verbindung existiert)
- Keine räumliche Strukturannahme
- Input wird als flacher Vektor betrachtet
Konsequenz:
MLPs ignorieren räumliche Beziehungen. Ein Pixel an Position (10,10) ist für das Modell nicht näher an (10,11) als an (200,200).

🔶 CNN – Faltungsschichten mit lokaler Wahrnehmung
Ein CNN nutzt:
- Convolution‑Kerne (Filter)
- Lokale Rezeptive Felder
- Parameter‑Sharing
- Pooling‑Operationen
Konsequenz:
CNNs erkennen Muster unabhängig von ihrer Position (Translation Invariance) und nutzen die räumliche Struktur von Bildern optimal aus.

📊 Vergleich CNN vs. MLP
![alt text](compare_mlp_cnn)


🧩 Beispiel: Warum CNNs für Bilder besser sind
Ein 64×64‑RGB‑Bild hat:
64\cdot 64\cdot 3=12,288\mathrm{\  Eingabewerte}
Ein MLP mit nur 100 Neuronen in der ersten Schicht hätte:
12,288\cdot 100=1,228,800\mathrm{\  Parameter}
Ein CNN mit einem 3×3‑Filter und 32 Kanälen hat:
3\cdot 3\cdot 3\cdot 32=864\mathrm{\  Parameter}
→ CNN: 864 Parameter vs. MLP: 1.2 Mio.
Das ist der Grund, warum CNNs Bilder so effizient verarbeiten.

🎮 RL‑Bezug (da du viel damit arbeitest)
- MLP: Perfekt für CartPole, MountainCar, LunarLander (Zustandsvektor)
- CNN: Pflicht für Atari, MuJoCo‑Kameras, Robotik‑Vision
DQN + CNN ist der Klassiker für Atari.

🧭 Kurzfassung in einem Satz
MLPs lernen Beziehungen zwischen Features ohne räumliche Struktur, während CNNs lokale Muster erkennen und räumliche Zusammenhänge ausnutzen – ideal für Bilder und visuelle RL‑Umgebungen.
