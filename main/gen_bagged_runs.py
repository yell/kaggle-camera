import env
from utils import RNG

a = [''] * 2 + ['-a']
b = ['-b 16'] * 3 + ['-b 12'] * 3 + ['-b 8']
cs = ['-cs 256'] * 2 + ['-cs 224']
d = ['-d 0', '-d 0.05', '-d 0.1', '-d 0.1', '-d 0.15'] # TODO
eu = ['-eu 8'] * 3 + ['-eu 6'] * 2 + ['-eu 4']
fc = [''] * 3 + ['-fc 1024 256'] * 2
l_lr = ['-lr 1e-4 1e-3 -clr 1e-3 4e-3'] * 5 + ['-l hinge -lr 1e-5 1e-4 -clr 1e-4 4e-4']
opt = [''] * 5 + ['-opt adam']
w = ['-w'] * 4 + ['']
nclr = [50] * 2 + [80] + [100] * 2
T = ['-t 0'] + ['-t 4'] * 5 + ['-t 6'] * 3 + ['-t 8'] * 2
dc = ['-dc 0.04', '-dc 0.08', '-dc 0.1']

seed = 111

rng = RNG(1337)
for i in xrange(100):
    tmpl = 'nohup python run.py {a} {b} {cs} {d} {eu} {fc} {opt} {w} -e 4000 -bt -rs {seed}'
    tmpl += ' {l_lr} {nclr} {T} {dc}'
    tmpl += ' -md ../models/{path}/ > {seed}.out &'
    current = {}
    for v in ('a', 'b', 'cs', 'd', 'eu', 'fc', 'opt', 'w', 'l_lr', 'nclr', 'T', 'dc'):
        current[v] = rng.choice(globals()[v])
    path = 'd121-bagg-{i}'.format(i=i + 1)
    # path = 'b{b}_cs{cs}_d{d}_eu{eu}_{fc}_{l}_{opt}_{w}_nclr{nclr}_seed{seed}'
    # path = path.format(b=current['b'][3:], cs=current['cs'][4:], d=current['d'][3:],
    #                    eu=current['eu'][4:], fc='fc' if current['fc'] else '', l=current['l'][3:],
    #                    opt='adam' if current['opt'] else '', w='w' if current['w'] else '',
    #                    nclr=current['nclr'], seed=seed)
    print tmpl.format(seed=seed, path=path, **current)
    seed += 111
