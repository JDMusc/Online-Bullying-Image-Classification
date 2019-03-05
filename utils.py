from toolz import pipe as p

def isImage(f):
    f = f.lower()
    return(any([e in f for e in ('jpg', 'jpeg', 'png')]))


def isOriginal(f):
    return(all([e not in f for e in ('flipped', 'noisy')]))


def unzip(tuple_arr):
    return p(zip(*tuple_arr), list)

