import env
from utils import RNG

b = ['-b 16', '-b 12', '-b 8']
cs = ['-cs 256', '-cs 256', '-cs 224', '-cs 224', '-cs 192']
d = ['-d 0', '-d 0.05', '-d 0.1', '-d 0.1', '-d 0.15']
eu = ['-eu 8', '-eu 8', '-eu 8', '-eu 6', '-eu 6', '-eu 4']
fc = ['', '-fc 1024 256']
l = ['-l logloss', '-l logloss', '-l logloss', '-l logloss', '-l hinge']
opt = ['', '', '', '', '-opt adam']
w = ['-w', '-w', '-w', '']
nclr = [50, 50, 50, 80, 100]

seed = 111

rng = RNG(1337)
for i in xrange(100):
    tmpl = 'nohup python run.py {b} {cs} {d} {eu} {fc} {l} {opt} {w} -e 4000 -bt -rs {seed}'
    tmpl += ' -lr 1e-4 1e-3 -clr 1e-3 4e-3 {nclr}'
    tmpl += ' -md ../models/{path}/ > {seed}.out &'
    current = {}
    for v in ('b', 'cs', 'd', 'eu', 'fc', 'l', 'opt', 'w', 'nclr'):
        current[v] = rng.choice(globals()[v])
    path = 'd121-bagg-{i}'.format(i=i + 1)
    # path = 'b{b}_cs{cs}_d{d}_eu{eu}_{fc}_{l}_{opt}_{w}_nclr{nclr}_seed{seed}'
    # path = path.format(b=current['b'][3:], cs=current['cs'][4:], d=current['d'][3:],
    #                    eu=current['eu'][4:], fc='fc' if current['fc'] else '', l=current['l'][3:],
    #                    opt='adam' if current['opt'] else '', w='w' if current['w'] else '',
    #                    nclr=current['nclr'], seed=seed)
    print tmpl.format(seed=seed, path=path, **current)
    seed += 111
