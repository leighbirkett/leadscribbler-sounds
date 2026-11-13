"""Rooms for LeadScribbler's Room control: Voxengo's IM Reverbs, by Aleksey Vaneev. Their licence lets them be passed
on free of charge only as complete, unaltered copies with its notice, so the files go into OUT/rooms/voxengo exactly as
they come, with license.txt beside them; the page itself shortens and levels the one it uses. sounds.json lists the
rooms the Room control offers under 'rooms'.
python rooms_build.py IMREVERBS_DIR OUT_DIR"""
import glob, json, os, shutil, sys

SRC, OUT = sys.argv[1], sys.argv[2]
PICK = {      # the Room control's room: (its name, the Voxengo file)
    'small': ('Small room', 'Nice Drum Room.wav'),
    'salon': ('Salon', 'French 18th Century Salon.wav'),
    'opera': ('Opera house', 'Scala Milan Opera Hall.wav'),
    'concert': ('Concert hall', 'Musikvereinsaal.wav'),
    'church': ('Church', 'St Nicolaes Church.wav'),
}
dst = os.path.join(OUT, 'rooms', 'voxengo'); os.makedirs(dst, exist_ok=True)
for f in glob.glob(os.path.join(SRC, '*')):
    shutil.copy2(f, dst)                         # every file, unaltered, and the licence
mpath = os.path.join(OUT, 'sounds.json'); man = json.load(open(mpath))
man['rooms'] = {k: dict(name=n, file='rooms/voxengo/' + fn, by='Voxengo IM Reverbs (Aleksey Vaneev): ' + fn[:-4]) for k, (n, fn) in PICK.items()}
json.dump(man, open(mpath, 'w'), separators=(',', ':'))
print(len(os.listdir(dst)), 'files in', dst); print(json.dumps(man['rooms'], indent=1))
