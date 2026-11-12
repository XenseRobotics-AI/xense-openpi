import jax
import pytest


@pytest.fixture(autouse=True, scope="session")
def _exact_matmuls():
    """Compare RLT computations in full float32.

    The equivalence tests (remat, batch independence, trimming, cache vs online features) assert
    ~1e-4. On GPUs whose default float32 matmul runs in TF32 or bf16 passes, kernel choice varies
    with batch shape and the results differ by a few 1e-4, so these tests failed intermittently.
    Set globally, not with the ``jax.default_matmul_precision`` context, which is thread-local
    and would miss the BackgroundTrainer thread.
    """
    previous = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_default_matmul_precision", previous)
