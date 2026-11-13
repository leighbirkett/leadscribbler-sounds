# LeadScribbler sounds

Real instruments for LeadScribbler's playback, every note recorded soft and loud, and real rooms to hear them in.
They're made small enough for a page to download only the notes its music uses, the first time it plays them (about
30 KB a note).

## Instruments

| Instrument | Recorded as | Dynamics | Source |
|---|---|---|---|
| Nylon-string guitar | string by string, E2–B5 | fingers, light pick, firm pick | Meta Open Sound |
| Steel-string guitar | string by string, E2–C6 | soft, medium, hard | Meta Open Sound |
| Upright piano | every key, A0–C8 | pp, mp, mf, f | Meta Open Sound |
| Violin | bowed, string by string | pp, mf, ff | Iowa |
| Violin, plucked | pizzicato, string by string | pp, mf, ff | Iowa |
| Viola | bowed, string by string | pp, mf, ff | Iowa |
| Cello | bowed, string by string | pp, mf, ff | Iowa |
| Double bass, bowed | string by string, with the low C extension | pp, mf, ff | Iowa |
| Double bass, plucked | pizzicato, string by string | pp, mf, ff | Iowa |
| Flute | with vibrato | pp, mf, ff | Iowa |
| Oboe | | pp, mf, ff | Iowa |
| Clarinet | B♭ clarinet | pp, mf, ff | Iowa |
| Bassoon | | pp, mf, ff | Iowa |
| French horn | | pp, mf, ff | Iowa |
| Trumpet | B♭ trumpet, without vibrato | pp, mf, ff | Iowa |
| Trombone | tenor trombone | pp, mf, ff | Iowa |
| Tuba | | pp, mf, ff | Iowa |
| Vibraphone | with the pedal down | pp, mf, ff | Iowa |
| Marimba | yarn mallets | pp, mf, ff | Iowa |

## Rooms

`rooms/voxengo` holds Voxengo's IM Reverbs, the recordings of how real spaces answer a click that LeadScribbler's Room
control plays everything through (convolution). They are exactly as Voxengo publishes them, with `license.txt` beside
them. LeadScribbler offers five: Nice Drum Room (as "Small room"), French 18th Century Salon, Scala Milan Opera Hall,
Musikvereinsaal (as "Concert hall") and St Nicolaes Church.

## Where they come from

- **Meta Open Sound** (github.com/facebookincubator/Meta-open-sound): the guitars and the piano. Dedicated to the public
  domain (CC0 1.0).
- **University of Iowa Electronic Music Studios, Musical Instrument Samples** (theremin.music.uiowa.edu/MIS.html): the
  strings, woodwinds, brass, vibraphone and marimba, recorded in the university's anechoic chamber. The university says
  they "may be downloaded and used for any projects, without restrictions".
- **Voxengo IM Reverbs** by Aleksey Vaneev (voxengo.com/impulses): the rooms. Free to use for any purpose; shared here
  free of charge, complete and unaltered, with Voxengo's copyright notice and conditions. Aleksey Vaneev keeps all rights
  in them.

Their terms are in `licences/` (and Voxengo's in `rooms/voxengo/license.txt`). The recordings are theirs; this collection
only prepares the instruments for the web. `iowa_split.py` cuts the Iowa recordings' chromatic runs into single notes
(taking out the anechoic chamber's low rumble), `build_sounds.py` trims each note to its attack, brings it into tune,
gives held notes a seamless loop and saves it as a small mono MP3, and `levels.py` measures how loud every note sounds
(in LUFS, over its first moments) and works out the gain that puts it at the loudness of the dynamic it stands for: a
phrase stays even from string to string and register to register, every instrument sits at the same level at mf (about
that of the General MIDI sounds, so the two mix evenly), and a step softer or louder is the same step on all of them.

The instruments are shared on the same terms as their sources: the Meta files are CC0 1.0; the Iowa files are free to
use for any project, without restrictions.

## Layout

`sounds.json` lists every file. For each instrument: its name, kind (`pluck`, `piano` or `sustain`), its dynamic layers,
the open strings (for the guitars and the bowed strings), and one row per file:

    [string zone, MIDI note, layer, attack (ms), loop start, loop end]

Files are `<instrument>/<zone>-<midi>-<layer>.mp3`. Zones are the strings, 0 = the lowest; instruments without strings
have one zone, 0. Loop points are in samples at 44.1 kHz.

Each instrument also has `fg`, the gain in dB to play each file at (keyed `zone-midi-layer`), and `lg`, the usual gain
for each layer. A player chooses the layer from how hard the note is played (0 to 1, about 0.6 for mf): two layers split
at 0.45; three at 0.42 and 0.7; four at 0.3, 0.5 and 0.7. The gains are set for a player that also plays louder within a
layer as a note is played harder; between them, soft to loud spans about 25 dB.

`rooms` names each room the Room control offers and the file it comes from.
