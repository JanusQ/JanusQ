## decomposition of unitary matrices
Let $U = \sum_{k=1}^{N} \mu_k U_k$ be a unitary matrix, where $\mu_k$ are the coefficients and $U_k$ are the unitary matrices. The decomposition of $U$ can be obtained by the following steps:
For a process $V_1(V_2\otimes U) V_3$, where $V_1$ and $V_3$ are unitary matrices, $V_2$ is a unitary matrix and $U$ is a unitary matrix, we have the following decomposition:
$V_1(V_2\otimes U) V_3 = \sum_{k=1}^{N} \mu_k (V_1 (V_2\otimes U_k) V_3)$.

To build the coefficients $\mu_k$ and the unitary matrices $U_k$, we can use the following method:
suppose the input state is $|\phi \rangle$ and the output state is $|\psi \rangle$, we have $|\psi \rangle = V_1(V_2\otimes U) V_3 |\phi \rangle$.
When we replace the $U$ with $U_k$, the output state is $|\psi_k \rangle = V_1(V_2\otimes U_k) V_3 |\phi \rangle$.

so we have $|\psi \rangle = \sum_{k=1}^{N} \mu_k |\psi_k \rangle$.

To find the coefficients $\mu_k$, we can use the following method:
$\mu_k = \langle \psi_k | \psi \rangle$.
If the state $| \psi_k\rangle$ is orthogonal, which means $\langle \psi_i | \psi_j \rangle = 0$ for $i \neq j$, that implies $\langle \psi_i | \psi_j \rangle = \langle \phi | V_3^\dagger (V_2^\dagger \otimes U_i^\dagger) V_1^\dagger V_1 (V_2\otimes U_j) V_3 |\phi \rangle = \langle V_3 \phi | (I\otimes U_i^\dagger U_j) | V_3 \phi \rangle = 0$.
Therefore, we have $U_i^\dagger U_j = 0$ for $i \neq j$. However, this can not be true because $U_i$ and $U_j$ are unitary matrices. Therefore, the state $| \psi_k\rangle$ is not orthogonal.

But the measurements on the state $| \psi_k\rangle$ and $|\psi \rangle$ has following relation:
let $|m\rangle$ be the measurement basis, we have $P_{\psi}(m) = \langle \psi |m \rangle \langle m | \psi \rangle$.
Therefore, we have $P_{\psi}(m) = \langle \sum_{k=1}^{N} \mu_k \psi_k|m\rangle \langle m |\sum_{k=1}^{N} \mu_k |\psi_k\rangle = \sum_{i=1}^{N} \mu^*_i \sum_{j=1}^{N} \mu_j \langle \psi_i |m\rangle \langle m |\psi_j\rangle$. But the measurement basis is orthogonal, so we have $P_{\psi}(m) = \sum_{i=1}^{N} |\mu_i|^2 P_{\psi_i}(m)$.




