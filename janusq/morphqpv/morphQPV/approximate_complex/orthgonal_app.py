import numpy as np
def decompose(input_state, base_states):
    ## calculate the parameters with inner product
    # parms = SgdOptimizer().optimize_params(base_states,input_state)
    parms =[]
    for i in range(len(base_states)):
        parms.append(input_state.dot(base_states[i]))
    return parms

def compose(parms,states):
    ## compose the state with parameters and base_states
    build_state = np.zeros(states[0].shape,dtype=np.complex128)
    for i in range(len(parms)):
        build_state += parms[i]*states[i]
    return build_state


def approximate(input_state, base_states):
    """ 通过基态构建量子态
    Args:
        input_state: 输入的量子态
        base_states: 基态
    Returns:
        build_state: 构建的量子态
    """
    bastparms = decompose(input_state, base_states)
    build_state = compose(bastparms, base_states)
    ## normalize the state
    build_state = build_state/np.linalg.norm(build_state,ord=2)
    return build_state