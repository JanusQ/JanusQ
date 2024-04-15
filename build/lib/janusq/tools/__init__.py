from typing import Iterable
from tqdm import tqdm


def pb(iterator: Iterable, show: bool):
    '''
        progress_bar
    '''
    if show:
        for elm in tqdm(iterator):
            yield elm
    else:
        for elm in iterator:
            yield elm