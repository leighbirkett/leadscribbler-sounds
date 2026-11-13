"""How loud each recording plays: every note brought to the loudness of the dynamic it stands for, so a phrase is even
from string to string and register to register, and every instrument sits at the same level at mf.
python levels.py OUT [base LUFS, default -32: about the level of the General MIDI sounds, so the two mix evenly]
Measures each file (LUFS over a note's first 0.6 s, or 0.2-1.2 s of a held one) and writes into sounds.json, per
instrument: 'fg', a gain in dB for each file ('zone-midi-layer'), and 'lg', the median for each layer (for any file
without one). A layer's target is base + 8 dB x (its dynamic - mf): playing harder adds the rest."""
import json, subprocess, sys, numpy as np, pyloudnorm as pyln
OUT = sys.argv[1]; BASE = float(sys.argv[2]) if len(sys.argv) > 2 else -32.0
man = json.load(open(f'{OUT}/sounds.json')); meter = pyln.Meter(44100, block_size=0.4)
TH = {2: [0.45], 3: [0.42, 0.7], 4: [0.3, 0.5, 0.7]}
def dec(f):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', f, '-f', 'f32le', '-ac', '1', '-ar', '44100', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)
for k, I in man['inst'].items():
    L = len(I['layers']); th = [0.0] + TH.get(L, []) + [1.0]
    centres = [(th[i] + th[i + 1]) / 2 for i in range(L)]
    fg, lg = {}, []
    for l in range(L):
        target = BASE + 8 * (centres[l] - 0.6) - (2 if k == 'bass' else 0)
        gains = {}
        for r in (r for r in I['s'] if r[2] == l):
            x = dec(f'{OUT}/{k}/{r[0]}-{r[1]}-{r[2]}.mp3'); on = int((r[3] or 4) / 1000 * 44100)
            seg = x[on + int(0.2 * 44100):on + int(1.2 * 44100)] if I['kind'] == 'sustain' else x[on:on + int(0.6 * 44100)]
            if len(seg) < int(0.45 * 44100): continue
            v = meter.integrated_loudness(seg)
            if np.isfinite(v): gains[f'{r[0]}-{r[1]}-{r[2]}'] = target - v
        med = float(np.median(list(gains.values()))) if gains else 0.0
        for f, g in gains.items(): fg[f] = round(max(med - 12, min(med + 12, g)), 1)
        lg.append(round(med, 1))
    I['fg'], I['lg'] = fg, lg
    print(k, 'layer gains dB', lg, 'spread', round(min(fg.values()), 1), 'to', round(max(fg.values()), 1))
json.dump(man, open(f'{OUT}/sounds.json', 'w'), separators=(',', ':'))
