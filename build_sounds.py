"""LeadScribbler sounds: free (CC0) multi-sampled instruments, made small enough for a web page to fetch a note at a time.

Sources:
  Meta Open Sound (github.com/facebookincubator/Meta-open-sound), CC0 1.0: nylon and steel-string guitars, sampled
    string by string with three dynamic layers; an upright piano with four.
  University of Iowa Electronic Music Studios, Musical Instrument Samples (theremin.music.uiowa.edu/MIS.html), free to
    use for any project without restrictions: violin (bowed and plucked) and cello, string by string, and flute, each
    at pp, mf and ff, recorded in an anechoic chamber. iowa_split.py cuts their chromatic runs into single notes first.
  VSCO 2 Community Edition by Versilian Studios (github.com/sgossner/VSCO-2-CE), CC0 1.0: the double bass (pizzicato).

Each sample becomes one mono MP3: trimmed to its attack, levelled within its instrument (keeping the real difference
between soft and loud), cut to a sensible length with a fade, and, for bowed and blown notes, given a seamless loop so
a held note can last as long as the music asks. sounds.json lists them all.

python build_sounds.py OUT_DIR
"""
import json, os, re, subprocess, sys, tempfile, glob
import numpy as np
import soundfile as sf

META = 'meta-open-sound/systems/instruments'
VSCO = 'VSCO-2-CE'
IOWA = 'iowa_notes'          # from iowa_split.py
OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else None
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def note_midi(name):
    """'Fs2', 'F#2', 'A#0', 'Bb3' -> MIDI (C4 = 60)."""
    m = re.match(r'^([A-G])(s|#|b)?(-?\d)$', name)
    if not m:
        raise ValueError(name)
    pc = PC[m.group(1)] + {'s': 1, '#': 1, 'b': -1, None: 0}[m.group(2)]
    return 12 * (int(m.group(3)) + 1) + pc


def read_mono(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    return x.mean(axis=1), sr


def f0_of(x, sr, lo=30.0, hi=2500.0):
    """The pitch of a steady stretch (YIN's difference function, first dip under the threshold)."""
    n = int(0.08 * sr)
    best = []
    for start in (0.25, 0.45, 0.65):
        s = int(start * sr)
        seg = x[s:s + 2 * n]
        if len(seg) < 2 * n:
            continue
        seg = seg - seg.mean()
        tmin, tmax = int(sr / hi), min(n - 1, int(sr / lo))
        d = np.array([np.sum((seg[:n] - seg[t:t + n]) ** 2) for t in range(tmax + 1)])
        cm = d[1:] * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)
        cm = np.concatenate([[1.0], cm])
        t = None
        for k in range(tmin, tmax):
            if cm[k] < 0.15 and cm[k] <= cm[k + 1]:
                t = k
                break
        if t is None:
            t = tmin + int(np.argmin(cm[tmin:tmax]))
        # parabolic refinement
        if 0 < t < len(cm) - 1:
            a, b, c = cm[t - 1], cm[t], cm[t + 1]
            den = a - 2 * b + c
            t = t + (0.5 * (a - c) / den if den else 0)
        best.append(sr / t)
    return float(np.median(best)) if best else None


def onset_of(x, sr):
    """The first moment the note is really there: the envelope's first rise past 3% of its peak."""
    env = np.abs(x)
    w = max(1, int(0.002 * sr))
    env = np.convolve(env, np.ones(w) / w, mode='same')
    pk = env.max() or 1.0
    i = int(np.argmax(env > 0.03 * pk))
    return i


def bake_loop(x, sr, on, kind, short=False):
    """A seamless loop in the sustain: the stretch from s to e plays round and round; the last X before e is crossfaded
    with the X before s, so the jump from e back to s is inaudible. Returns (audio up to e, s, e) in samples.
    short: for notes of only two seconds or so (the Iowa flute), a shorter loop nearer the attack."""
    n = len(x)
    a0 = on + int((0.45 if short else 0.9) * sr)      # past the attack
    a1 = min(n - int((0.25 if short else 0.3) * sr), on + int(4.5 * sr))   # well before the note's own release
    if a1 - a0 < int((0.65 if short else 1.2) * sr):
        return None
    L0, L1 = (int(0.4 * sr), int(0.9 * sr)) if short else (int(0.8 * sr), int(1.6 * sr))
    # coarse search on an energy envelope, then the waveform itself around the best match
    hop = int(0.005 * sr)
    env = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode='same'))[::hop]
    best = None
    for L in range(L0 // hop, L1 // hop):
        for s in range(a0 // hop, (a1 // hop) - L):
            e = s + L
            w = int(0.06 * sr) // hop
            if e + w >= len(env) or s - w < 0:
                continue
            da = env[s - w:s + w]
            db = env[e - w:e + w]
            c = np.sum(np.abs(da - db)) / (np.sum(da) + 1e-9)
            if best is None or c < best[0]:
                best = (c, s * hop, e * hop)
    if not best:
        return None
    _, s, e = best
    # fine: shift the end by up to a pitch period to match the waveform at the start
    win = int(0.02 * sr)
    ref = x[s:s + win]
    span = int(0.012 * sr)
    cands = range(max(e - span, s + L0 // 2), min(e + span, n - win))
    e = max(cands, key=lambda k: float(np.dot(ref, x[k:k + win])))
    X = int(0.12 * sr)
    X = min(X, s - on - int(0.05 * sr))
    if X < int(0.03 * sr):
        return None
    # after the loop's end, the audio carries on as it would from the loop's start: the file's own last frames (where an
    # MP3 is least exact) are then never reached
    pad = int(0.2 * sr)
    y = np.concatenate([x[:e], x[s:s + pad]]).astype(np.float32)
    t = np.linspace(0, np.pi / 2, X, dtype=np.float32)
    fade_out, fade_in = np.cos(t), np.sin(t)
    y[e - X:e] = x[e - X:e] * fade_out + x[s - X:s] * fade_in
    f = int(0.05 * sr); y[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    return y, s, e


def decoded_onset(path):
    """Where a browser will find the attack in the MP3, the way the page looks for it (the first sample past 3% of the
    first half-second's peak), in ms: the page compares its own reading with this, to allow for decoders that pad."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', '44100', '-'], check=True,
                         capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    pk = float(np.abs(x[:22050]).max()) or 1.0
    return round(int(np.argmax(np.abs(x) > 0.03 * pk)) / 44.1, 1)


def encode(y, sr, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t:
        sf.write(t.name, y, sr, subtype='FLOAT')
        tmp = t.name
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-ac', '1', '-ar', '44100', '-c:a', 'libmp3lame', '-q:a', '3', out],
                       check=True)
    finally:
        os.unlink(tmp)


# ---------- the instruments ----------
STR6 = ['LowE', 'A', 'D', 'G', 'B', 'HighE']
INST = {
    'nylon': dict(id=200, gm=24, name='Nylon-string guitar', kind='pluck', strings=[40, 45, 50, 55, 59, 64],
                  layers=['fingers', 'light pick', 'firm pick'], maxlen=5.0, src='meta'),
    'steel': dict(id=201, gm=25, name='Steel-string guitar', kind='pluck', strings=[40, 45, 50, 55, 59, 64],
                  layers=['soft', 'medium', 'hard'], maxlen=6.0, src='meta'),
    'piano': dict(id=202, gm=0, name='Upright piano', kind='piano', layers=['pp', 'mp', 'mf', 'f'], maxlen=6.0, src='meta'),
    'violin': dict(id=203, gm=40, name='Violin', kind='sustain', strings=[55, 62, 69, 76], layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'flute': dict(id=204, gm=73, name='Flute', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'cello': dict(id=205, gm=42, name='Cello', kind='sustain', strings=[36, 43, 50, 57], layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'bass': dict(id=206, gm=32, name='Double bass, plucked', kind='pluck', strings=[24, 28, 33, 38, 43], layers=['pp', 'mf', 'ff'], maxlen=3.0, src='iowa'),
    'pizz': dict(id=207, gm=45, name='Violin, plucked', kind='pluck', strings=[55, 62, 69, 76], layers=['pp', 'mf', 'ff'], maxlen=2.0, src='iowa'),
    'viola': dict(id=208, gm=41, name='Viola', kind='sustain', strings=[48, 55, 62, 69], layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'contrabass': dict(id=209, gm=43, name='Double bass, bowed', kind='sustain', strings=[24, 28, 33, 38, 43], layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'oboe': dict(id=210, gm=68, name='Oboe', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'clarinet': dict(id=211, gm=71, name='Clarinet', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'bassoon': dict(id=212, gm=70, name='Bassoon', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'horn': dict(id=213, gm=60, name='French horn', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'trumpet': dict(id=214, gm=56, name='Trumpet', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'trombone': dict(id=215, gm=57, name='Trombone', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'tuba': dict(id=216, gm=58, name='Tuba', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'altosax': dict(id=217, gm=65, name='Alto saxophone', kind='sustain', layers=['pp', 'mf', 'ff'], maxlen=4.6, src='iowa'),
    'vibraphone': dict(id=218, gm=11, name='Vibraphone', kind='piano', layers=['pp', 'mf', 'ff'], maxlen=6.0, src='iowa'),
    'marimba': dict(id=219, gm=12, name='Marimba', kind='pluck', layers=['pp', 'mf', 'ff'], maxlen=2.5, src='iowa'),
}


def sources(key):
    """[(zone, named midi, layer, path)]"""
    out = []
    if INST[key]['src'] == 'iowa':
        for p in glob.glob(f'{IOWA}/{key}/*.wav'):
            z, m, l = map(int, os.path.basename(p)[:-4].split('-'))
            out.append((z, m, l, p))
        return out
    if key == 'nylon':
        lay = {'Vel01Finger': 0, 'Vel02Pick': 1, 'Vel03Pick': 2}
        for p in glob.glob(f'{META}/meta-nylon-guitar/ogg/*.ogg'):
            m = re.match(r'INSGuitarNylon_String(\w+?)_(Vel0\d\w+)_([A-G]s?\d)_01\.ogg', os.path.basename(p))
            out.append((STR6.index(m.group(1)), note_midi(m.group(3)), lay[m.group(2)], p))
    elif key == 'steel':
        for p in glob.glob(f'{META}/meta-steel-guitar/ogg/*.ogg'):
            m = re.match(r'INSGuitarSteel_String(\w+?)_Vel0(\d)_([A-G]s?\d)_01\.ogg', os.path.basename(p))
            out.append((STR6.index(m.group(1)), note_midi(m.group(3)), int(m.group(2)) - 1, p))
    elif key == 'piano':
        for p in glob.glob(f'{META}/meta-piano/ogg/*.ogg'):
            m = re.match(r'INSPiano_Vel0(\d)_([A-G]s?\d)_01\.ogg', os.path.basename(p))
            out.append((0, note_midi(m.group(2)), int(m.group(1)) - 1, p))
    elif key == 'violin':
        for p in glob.glob(f'{VSCO}/Strings/Solo Violin/Arco Vib/*.wav'):
            m = re.match(r'LLVln_ArcoVib_([A-G]#?\d)_([pf])\.wav', os.path.basename(p))
            out.append((0, note_midi(m.group(1)), 0 if m.group(2) == 'p' else 1, p))
    elif key == 'pizz':
        for p in glob.glob(f'{VSCO}/Strings/Solo Violin/Pizz/*_RR1.wav'):
            m = re.match(r'LLVln_Pizz_([A-G]#?\d)_([pf])_RR1\.wav', os.path.basename(p))
            out.append((0, note_midi(m.group(1)), 0 if m.group(2) == 'p' else 1, p))
    elif key == 'flute':
        for p in glob.glob(f'{VSCO}/Woodwinds/Flute/susNV/*_v1_1.wav'):
            m = re.match(r'LDFlute_susNV_([A-G]#?\d)_v1_1\.wav', os.path.basename(p))
            out.append((0, note_midi(m.group(1)), 0, p))
        for p in glob.glob(f'{VSCO}/Woodwinds/Flute/susvib/*_v1_1.wav'):
            m = re.match(r'LDFlute_susvib_([A-G]#?\d)_v1_1\.wav', os.path.basename(p))
            out.append((0, note_midi(m.group(1)), 1, p))
    elif key == 'cello':
        for p in glob.glob(f'{VSCO}/Strings/Cello Section/susvib/*_1.wav'):
            m = re.match(r'susvib_([A-G]#?\d)_v([13])_1\.wav', os.path.basename(p))
            out.append((0, note_midi(m.group(1)), 0 if m.group(2) == '1' else 1, p))
    elif key == 'bass':
        for p in glob.glob(f'{VSCO}/Strings/Solo Contrabass/Pizz/*_rr1.wav'):
            m = re.match(r'BKCtbss_Pizz_([A-G]#?\d)_v([13])_rr1\.wav', os.path.basename(p))
            out.append((0, note_midi(m.group(1)), 0 if m.group(2) == '1' else 1, p))
    return out


def main():
    man = {'v': 1, 'inst': {}}
    mpath = os.path.join(OUT, 'sounds.json')
    if ONLY and os.path.exists(mpath):
        man = json.load(open(mpath))
    for key, I in INST.items():
        if ONLY and key not in ONLY:
            continue
        rows = []
        S = sources(key)
        print(key, len(S), 'samples', flush=True)
        # the instrument's own octave naming: check it against the pitch actually heard (VSCO numbers its octaves
        # differently from instrument to instrument), on a few samples
        shift = 0
        if I['src'] == 'vsco':
            votes = []
            for z, mid, l, p in sorted(S, key=lambda r: r[1])[::max(1, len(S) // 6)]:
                x, sr = read_mono(p)
                on = onset_of(x, sr)
                f = f0_of(x[on:], sr)
                if f:
                    heard = 69 + 12 * np.log2(f / 440.0)
                    votes.append(round((heard - mid) / 12) * 12)
            shift = int(max(set(votes), key=votes.count)) if votes else 0
            print('  octave correction', shift, votes, flush=True)
        cents = {}
        if I['src'] == 'iowa':
            rp = os.path.join(IOWA, f'report_{key}.json')
            rep = json.load(open(rp if os.path.exists(rp) else os.path.join(IOWA, 'report.json')))[key]
            cents = {k: v['cents'] for k, v in rep.items() if v.get('cents')}
        loaded = []
        for z, mid, l, p in S:
            x, sr = read_mono(p)
            c = cents.get(f'{z}-{mid}-{l}')
            if c and abs(c) >= 3:        # sharp by c cents: stretched by 2^(c/1200), so it plays in tune
                from fractions import Fraction
                from scipy.signal import resample_poly
                fr = Fraction(2 ** (c / 1200)).limit_denominator(4000)
                x = resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
            on = onset_of(x, sr)
            a = max(0, on - int(0.004 * sr))
            x = x[a:]
            loaded.append((z, mid + shift, l, x, sr, on - a))
        # level: the loudest layer's typical peak to -4 dBFS; every sample keeps its place relative to the others
        top = len(I['layers']) - 1
        pk_top = [np.max(np.abs(x)) for z, m, l, x, sr, on in loaded if l == top] or [np.max(np.abs(x)) for *_, x, sr, on in loaded]
        gain = 0.63 / float(np.percentile(pk_top, 75))
        for z, mid, l, x, sr, on in loaded:
            y = x * gain
            pk = np.max(np.abs(y))
            if pk > 0.97:
                y = y * (0.97 / pk)
            row = [z, mid, l, 0]
            if I['kind'] == 'sustain':
                lp = bake_loop(y, sr, on, I['kind']) or bake_loop(y, sr, on, I['kind'], short=True)
                if lp:
                    y, s, e = lp
                    row += [int(round(s * 44100 / sr)), int(round(e * 44100 / sr))]      # in samples at 44.1 kHz: exact, so the loop closes on itself
                else:
                    n = min(len(y), int(I['maxlen'] * sr))
                    y = y[:n].copy(); f = int(0.25 * sr); y[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
            else:
                n = min(len(y), int(I['maxlen'] * sr))
                y = y[:n].copy()
                f = min(int(0.6 * sr), n // 3)
                y[-f:] *= np.linspace(1, 0, f, dtype=np.float32) ** 2
            out = os.path.join(OUT, key, f'{z}-{mid}-{l}.mp3')
            encode(y, sr, out)
            row[3] = decoded_onset(out)
            rows.append(row)
        rows.sort()
        man['inst'][key] = {k: I[k] for k in ('id', 'gm', 'name', 'kind', 'layers')}
        if 'strings' in I:
            man['inst'][key]['strings'] = I['strings']
        man['inst'][key]['s'] = rows          # [string zone, MIDI, layer, attack (ms), loop start, loop end (samples at 44.1 kHz)]
        sz = sum(os.path.getsize(os.path.join(OUT, key, f)) for f in os.listdir(os.path.join(OUT, key)))
        print(f'  {len(rows)} files, {sz/1e6:.1f} MB, loops {sum(1 for r in rows if len(r) > 4)}', flush=True)
        man['about'] = ABOUT
        json.dump(man, open(mpath, 'w'), separators=(',', ':'))
    print('done', mpath)


ABOUT = ('LeadScribbler sounds. Meta Open Sound (guitars, piano; CC0); University of Iowa Electronic Music Studios, '
         'Musical Instrument Samples (violin, cello, flute; free to use without restrictions); VSCO 2 Community Edition by '
         'Versilian Studios (double bass; CC0).')


if __name__ == '__main__':
    main()
