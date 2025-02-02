import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import jax


@partial(vmap, in_axes=(None, 0, None))
def get_row(ops: jnp.ndarray, inp_ind: jnp.ndarray,
            out_ind: jnp.ndarray,) -> jnp.ndarray:
    """
    With K' tracked bitstrings and N qubits, this function returns a given row
    of the K x K' sub-matrix corresponding to the tracked bitstrings of the full
    2^N x 2^N matrix stored as a kronecker product of N matrices. out_ind
    specifies a row of the K x K' matrix, and inp_ind specifies the bitstrings
    corresponding to each column. inp_ind selects the relevant entries from
    the N single-qubit measurement error matrices for a given column and the
    function multiplies them together. JAX function vmap is used to vectorize
    this computation over all columns for a given row.

    :param inp_ind: a K' x N jax ndarray where each row encodes the tracked
                    bitstring corresponding to a given column
    :param out_ind: an N-dimensional 0/1 jax array encoding a tracked bitstring
    :param ops: an N x 2 x 2 jax ndarray of single-qubit error matrices
    :return: a K' x 1 row of the K x K' sub-matrix of the 2^N x 2^N error matrix
    """
    return jnp.prod(ops[jnp.arange(inp_ind.shape[0]), out_ind, inp_ind])


@jit
@partial(vmap, in_axes=(None, None, None, 0))
def fast_kron_matmul(ops: jnp.ndarray, state: jnp.ndarray,
                     inp_inds: jnp.ndarray,
                     out_inds: jnp.ndarray) -> jnp.ndarray:
    """
    This function multiplies ops @ state, where ops are the single-qubit
    Kronecker product decomposition of the full 2^N x 2^N measurement error
    matrix.
    In practice, XLA constructs the full matrix first and then does the matrix
    multiplication all at once for speed. If there are memory issues, consider
    using _kron_matmul_col_first(...) (preferred) or _kron_matmul_row_first
    (slower).

    :param ops: an N x 2 x 2 jax ndarray of single-qubit error matrices; the
                kronecker product decomposition of the full 2^N x 2^N
                measurement error matrix
    :param state: the K'-dimensional vector to be right-multiplied
    :param inp_inds: a K' x N jax ndarray where each row encodes the tracked
                    bitstring corresponding to a given column of the full
                    2^N x 2^N measurement error matrix
    :param out_inds: a K x N jax ndarray where each row encodes the tracked
                    bitstring corresponding to a given row. May be different
                    from inp_inds.
    :return: a K-dimensional vector
    """
    return jnp.dot(get_row(ops, inp_inds, out_inds), state)


@jit
def _kron_matmul_row_first(ops: jnp.ndarray, state: jnp.ndarray,
                           inp_inds: jnp.ndarray,
                           out_inds: jnp.ndarray) -> jnp.ndarray:
    """
        A memory efficient (but slower) implementation of ops @ state, where the
        ops are a Kronecker product decomposition. The rows are computed in
        parallel with vmapped get_row(...), and lax.map takes in each row of
        out_inds, produces the corresponding row with get_row (vmapped) and
        dots it with state, and stacks them to produce the solution.

        See fast_kron_matmul(...) for info on inputs.
    """
    return jax.lax.map(lambda x: jnp.dot(get_row(ops, inp_inds, x), state),
                       out_inds)


@partial(vmap, in_axes=(None, None, 0))
def get_col(ops: jnp.ndarray, inp_ind: jnp.ndarray,
            out_ind: jnp.ndarray) -> jnp.ndarray:
    """
        Exactly analogous to get_row(...), except this is vmapped over the
        out_inds instead of inp_inds, so it constructs a given column of the
        matrix stored as a decomposition in ops. See get_row(...) for more info
        on inputs.
    """
    return jnp.prod(ops[jnp.arange(inp_ind.shape[0]), out_ind, inp_ind])


@jit
def compact_kron_matmul(ops: jnp.ndarray,
                        state: jnp.ndarray,
                        inp_inds: jnp.ndarray,
                        out_inds: jnp.ndarray) -> jnp.ndarray:
    """
        A memory efficient (but slower) implementation of ops @ state, where the
        ops are a Kronecker product decomposition. lax.scan takes in each row,
        computes the corresponding column in parallel with vmapped get_col(...),
        multiplies it by the corresponding entry in the input vector state,
        and adds this to the running sum.

        See fast_kron_matmul(...) for info on inputs.
        """
    def scanner(carry, col_and_elem):
        col = col_and_elem[:col_and_elem.shape[0] - 1].astype(int)
        elem = col_and_elem[col_and_elem.shape[0] - 1]
        return carry + get_col(ops, col, out_inds).reshape(-1, 1) * elem, None

    return jax.lax.scan(scanner, jnp.zeros([out_inds.shape[0], 1]),
                        jnp.hstack([inp_inds, state]))[0]
