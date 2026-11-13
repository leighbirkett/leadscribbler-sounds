"""University of Iowa Musical Instrument Samples: their recordings are chromatic runs, one file per string and dynamic
(Violin.arco.mf.sulG.G3B3.mono.aif) with each note separated by silence. This cuts them into one file per note,
named <zone>-<midi>-<layer>.wav, for build_sounds.py.

The anechoic chamber's only noise is a rumble below the instruments' lowest notes, so each run is high-passed first.
Transposing instruments (clarinet, trumpet, horn, saxophone) are named at written pitch: the pitch heard decides.
python iowa_split.py SRC_DIR OUT_DIR [instrument,instrument...]"""
import glob, json, os, re, sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SRC, OUT = sys.argv[1], sys.argv[2]
ONLY = sys.argv[3].split(',') if len(sys.argv) > 3 else None
PLUCK = {'pizz', 'bass', 'vibraphone', 'marimba'}       # struck or plucked: found from their attacks, heard just after
# the viola's '44.1 kHz' files are its 96 kHz recordings with the wrong rate in their headers (they sound 13.5
# semitones low and last twice as long): read at 96 kHz and brought down to 44.1
TRUE_SR = {'viola': 96000}
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
LAYER = {'pp': 0, 'mf': 1, 'ff': 2}
# instrument: (file glob, strings (zone order) or None, high-pass Hz, lowest and highest MIDI it plays)
INST = {
    'violin': ('Violin.arco.mono.1644.1/Violin.arco.*.aif', ['G', 'D', 'A', 'E'], 90, 55, 103),
    'pizz': ('Violin.pizz.mono.1644.1/Violin.pizz.*.aif', ['G', 'D', 'A', 'E'], 90, 55, 103),
    'cello': ('Cello.arco.mono.1644.1/Cello.arco.*.aif', ['C', 'G', 'D', 'A'], 40, 36, 81),
    'flute': ('flute.vib/flute.vib.*.aiff', None, 110, 59, 98),
    'viola': ('Viola.arco.mono.1644.1/Viola.arco.*.aif', ['C', 'G', 'D', 'A'], 60, 48, 93),       # really 96 kHz (see TRUE_SR)
    'contrabass': ('Bass.arco.mono.1644.1/Bass.arco.*.aif', ['C', 'E', 'A', 'D', 'G'], 22, 24, 67),
    'bass': ('Bass.pizz.mono.1644.1/Bass.pizz.*.aif', ['C', 'E', 'A', 'D', 'G'], 22, 24, 67),
    'oboe': ('oboe/oboe.*.aiff', None, 150, 57, 93),
    'clarinet': ('BbClar/BbClar.*.aiff', None, 100, 49, 94),
    'bassoon': ('Bassoon/Bassoon.*.aiff', None, 40, 33, 77),
    'horn': ('Horn/Horn.*.aiff', None, 40, 33, 79),
    'trumpet': ('Trumpet.novib/Trumpet.novib.*.aiff', None, 120, 52, 88),
    'trombone': ('TenorTrombone/TenorTrombone.*.aiff', None, 50, 38, 77),
    'tuba': ('tuba/tuba.*.aiff', None, 22, 24, 67),
    'altosax': ('AltoSax.Vib/AltoSax.Vib.*.aiff', None, 90, 47, 82),
    'vibraphone': ('Vibraphone/Vibraphone.sustain.*.aif', None, 60, 48, 91),
    'marimba': ('Marimba/Marimba.yarn.*.aif', None, 40, 36, 96),
}


def parse(base):
    """Its dynamic, string (or None) and range of notes, wherever they sit in the name."""
    toks = [t.strip() for t in base.split('.')]
    dyn = next((t for t in toks if t in LAYER), None)
    sul = next((t[3:] for t in toks if t.startswith('sul')), None)
    rng = next((t for t in toks if re.fullmatch(r'([A-G][b#]?\d){1,2}', t)), None)
    return dyn, sul, rng


def midi(name):
    m = re.match(r'([A-G])(b|#)?(\d)', name)
    return 12 * (int(m.group(3)) + 1) + PC[m.group(1)] + {'b': -1, '#': 1, None: 0}[m.group(2)]


def expected(rng):
    ns = re.findall(r'[A-G][b#]?\d', rng)
    a = midi(ns[0]); b = midi(ns[-1]) if len(ns) > 1 else a
    return list(range(a, b + 1))


def f0(x, sr, start, lo, hi, win=0.06, count=6):
    """Median pitch (YIN) over a few windows from 'start' on."""
    n = int(win * sr); out = []
    for k in range(count):
        s = start + int(k * 1.5 * win * sr); seg = x[s:s + 2 * n]
        if len(seg) < 2 * n:
            break
        seg = seg - seg.mean()
        tmin, tmax = int(sr / hi), min(n - 1, int(sr / lo))
        d = np.array([np.sum((seg[:n] - seg[t:t + n]) ** 2) for t in range(tmax + 1)])
        cm = np.concatenate([[1.0], d[1:] * np.arange(1, tmax + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)])
        t = next((k2 for k2 in range(tmin, tmax) if cm[k2] < 0.12 and cm[k2] <= cm[k2 + 1]), None)
        if t is None:
            continue
        a, b, c = cm[t - 1], cm[t], cm[t + 1]; den = a - 2 * b + c
        out.append(sr / (t + (0.5 * (a - c) / den if den else 0)))
    return float(np.median(out)) if out else None


def segments_env(y, sr, pluck):
    """For runs without digital silence between the notes: the notes stand well clear of the chamber's quiet, so each
    stretch more than 20 dB above it is a note (a dip of under a quarter of a second inside one doesn't count)."""
    h = int(0.01 * sr); n = len(y) // h
    e = 20 * np.log10(np.sqrt((y[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-12)
    floor = float(np.percentile(e[e > -200], 10)) if (e > -200).any() else -100
    th = min(floor + 20, float(e.max()) - 25)
    on = e > th; runs, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append([i, j]); i = j
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < 25:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        a0 = max(0, a - 30); eb = e[a0:b]; pk = float(eb.max())
        st = a0 + int(np.argmax(eb >= pk - 30))
        tail = e[st:min(n, b + 300)]; lim = max(pk - (55 if pluck else 45), floor + 8)
        above = np.where(tail >= lim)[0]
        end = st + int(above[-1]) + 1 if len(above) else b
        if (end - st) * h >= (0.15 if pluck else 0.5) * sr:
            out.append((st * h, end * h))
    return out


def segments_pitch(y, sr, lo_m, hi_m):
    """For runs played one note straight after another (the older bassoon and horn recordings): a note is a stretch
    where the pitch holds on one semitone for a third of a second or more. Each starts at the dip in loudness just
    before it (where the player tongued it), so it keeps its attack."""
    import librosa
    hop = 512
    f = librosa.yin(y.astype(np.float32), fmin=440 * 2 ** ((lo_m - 2.5 - 69) / 12), fmax=440 * 2 ** ((hi_m + 2.5 - 69) / 12),
                    sr=sr, frame_length=4096, hop_length=hop)
    m = 69 + 12 * np.log2(f / 440.0)
    rms = librosa.feature.rms(y=y.astype(np.float32), frame_length=2048, hop_length=hop)[0]
    e = 20 * np.log10(rms + 1e-12); n = min(len(m), len(e)); m, e = m[:n], e[:n]
    voiced = e > e.max() - 35          # (no silence to measure a floor from: the notes run on)
    r = np.where(voiced, np.round(m), -1)
    runs, i = [], 0
    while i < n:
        j = i
        while j < n and (r[j] == r[i] or (j + 2 < n and r[j + 1] == r[i] and r[j + 2] == r[i])):
            j += 1
        if r[i] >= 0 and (j - i) * hop >= 0.35 * sr:
            runs.append((i, j))
        i = max(j, i + 1)
    out = []
    for i, j in runs:
        k0 = max(0, i - int(0.15 * sr / hop)); st = k0 + int(np.argmin(e[k0:i + 1]))      # the tonguing dip
        out.append((st * hop, j * hop))
    return out


def segments_by(x, y, sr, pluck, nexp, lo_m=None, hi_m=None):
    """Whichever way of finding the notes comes closest to the number the file's name promises."""
    cands = [segments(x, y, sr, pluck), segments_env(y, sr, pluck)]
    if not pluck and lo_m is not None and min(abs(len(c) - nexp) for c in cands) > max(1, nexp // 4):
        cands.append(segments_pitch(y, sr, lo_m, hi_m))
    return min(cands, key=lambda c: abs(len(c) - nexp))


def segments(x, y, sr, pluck):
    """The editors left digital silence between the notes: each stretch between silences holds one note (and the
    chamber's faint noise around it). The note runs from where it comes within 30 dB of its peak to where it has died
    away to the noise."""
    z = np.abs(x) < 1e-7
    h = int(0.01 * sr); n = len(y) // h
    e = 20 * np.log10(np.sqrt((y[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-12)
    zf = z[:n * h].reshape(n, h).all(1)
    top = float(e.max())
    blocks, i = [], 0
    while i < n:
        if zf[i]:
            i += 1; continue
        j = i
        while j < n and (not zf[j] or (j + 2 < n and not zf[j + 1:j + 3].all())):
            j += 1
        blocks.append((i, j)); i = j
    out = []
    for a, b in blocks:
        eb = e[a:b]
        if len(eb) < 20:
            continue
        pk = float(eb.max())
        if pk < top - 45:
            continue
        noise = float(np.percentile(eb, 20))
        on = a + int(np.argmax(eb >= pk - 30))
        lim = max(pk - (55 if pluck else 45), noise + 8)
        above = np.where(eb >= lim)[0]
        if not len(above):
            continue
        end = a + int(above[-1]) + 1
        if (end - on) * h < (0.15 if pluck else 0.6) * sr:
            continue
        out.append((on * h, end * h))
    return out


def main():
    rpath = os.path.join(OUT, 'report.json')
    report = json.load(open(rpath)) if os.path.exists(rpath) and not ONLY else {}
    for inst, (pat, strings, hp, lo, hi) in INST.items():
        if ONLY and inst not in ONLY:
            continue
        os.makedirs(os.path.join(OUT, inst), exist_ok=True)
        got = {}
        files = []
        for p in sorted(glob.glob(os.path.join(SRC, pat))):
            base = os.path.basename(p)
            dyn, sul, rng = parse(base)
            if base.startswith('._') or dyn is None or rng is None or (strings and sul not in strings):
                continue
            try:
                x, sr = sf.read(p, always_2d=True, dtype='float64'); x = x.mean(1)
                if inst in TRUE_SR:
                    from scipy.signal import resample_poly
                    from fractions import Fraction
                    fr = Fraction(44100, TRUE_SR[inst]); x = resample_poly(x, fr.numerator, fr.denominator); sr = 44100
            except Exception as e:
                print('  could not read', base, e, flush=True); continue
            y = sosfiltfilt(butter(4, hp, 'hp', fs=sr, output='sos'), x)
            ex = expected(rng)
            segs = segments_by(x, y, sr, inst in PLUCK, len(ex), min(ex) - 10, max(ex) + 3)   # (a transposing instrument sounds up to 9 below its names)
            files.append([base, LAYER[dyn], strings.index(sul) if strings else 0, expected(rng), y, sr, segs, None])
        st0 = lambda a, sr: a + int((0.03 if inst in PLUCK else 0.35) * sr)
        win, cnt = (0.04, 4) if inst in PLUCK else (0.08, 10)
        # how far the names are from the notes heard: a transposing instrument is named at written pitch (a clarinet or
        # trumpet sounds a tone lower, a horn a fifth, an alto saxophone a major sixth). The first note of each run,
        # heard freely, votes; octaves don't count, since a low note is easily heard an octave out
        votes = []
        for base, layer, zone, exp, y, sr, segs, _ in files:
            if not segs:
                continue
            f = f0(y, sr, st0(segs[0][0], sr), 25, 3200, win, cnt)
            if f:
                h = 69 + 12 * np.log2(f / 440.0)
                if abs(h - round(h)) < 0.4:
                    votes.append({0: 0, 10: -2, 5: -7, 3: -9}.get((int(round(h)) - exp[0]) % 12))
        votes = [v for v in votes if v is not None]
        T = max(set(votes), key=votes.count) if votes else 0
        if T:
            print(f'{inst}: named {T:+d} semitones from the sound (written pitch)', flush=True)
        # then every note, listened for only around where its run lies, so it can't be heard an octave out
        hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
        for F in files:
            base, layer, zone, exp, y, sr, segs, _ = F
            lo_hz, hi_hz = hz(exp[0] + T - 2.5), hz(exp[-1] + T + 2.5)
            heard = []
            for a, b in segs:
                f = f0(y, sr, st0(a, sr), lo_hz, hi_hz, win, cnt)
                heard.append(None if not f else 69 + 12 * np.log2(f / 440.0))
            F[7] = heard
        for base, layer, zone, exp, y, sr, segs, heard in files:
            exp = [m + T for m in exp]
            names = []
            if len(segs) == len(exp) and all(h is None or abs(h - m) < 0.6 for h, m in zip(heard, exp)):
                names = exp
            else:
                # a chromatic run: each note a semitone or so above the last one kept, starting where the file's
                # name says; a pitch that jumps (a harmonic read instead of the note) is left out
                last = None
                for h in heard:
                    m = None if h is None or abs(h - round(h)) > 0.4 else int(round(h))
                    if m is not None and (abs(m - exp[0]) <= 2 if last is None else last < m <= last + 3):
                        last = m
                    else:
                        m = None
                    names.append(m)
            ok = 0
            for (a, b), m, h in zip(segs, names, heard):
                if m is None or not lo <= m <= hi:
                    continue
                key = (zone, m, layer)
                if key in got:
                    continue
                s0 = max(0, a - int(0.03 * sr)); e0 = min(len(y), b + int(0.15 * sr))
                seg = y[s0:e0].astype(np.float32)
                f = int(0.02 * sr); seg[:f] *= np.linspace(0, 1, f, dtype=np.float32)
                seg[-int(0.1 * sr):] *= np.linspace(1, 0, int(0.1 * sr), dtype=np.float32)
                sf.write(os.path.join(OUT, inst, f'{zone}-{m}-{layer}.wav'), seg, sr, subtype='PCM_24')
                got[key] = dict(src=base, cents=None if h is None else round(100 * (h - m)), secs=round((b - a) / sr, 2)); ok += 1
            print(f'{inst:6s} {base:42s} expected {len(exp):2d} segments {len(segs):2d} kept {ok:2d}'
                  + ('' if len(segs) == len(exp) else '  heard ' + ' '.join('-' if h is None else f'{h:.1f}' for h in heard)), flush=True)
        report[inst] = {f'{z}-{m}-{l}': v for (z, m, l), v in sorted(got.items())}
        cover = sorted({m for (z, m, l) in got}); lay = {l: sum(1 for k in got if k[2] == l) for l in range(3)}
        print(f'== {inst}: {len(got)} notes, MIDI {cover[0]}-{cover[-1]} ({len(cover)} pitches), by layer {lay}', flush=True)
    if ONLY:                 # one report for each instrument, so several can be cut at once
        for k, v in report.items():
            json.dump({k: v}, open(os.path.join(OUT, f'report_{k}.json'), 'w'), indent=1)
    else:
        json.dump(report, open(rpath, 'w'), indent=1)


if __name__ == '__main__':
    main()
