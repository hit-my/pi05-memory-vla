"""policy.py serving with action-sample averaging: same checkpoint, less sampling jitter.

Each query draws ACTION_SAMPLES chunks from the flow (independent noise) and executes their mean. Memory
generation is unchanged. Tests whether near-miss placements come from sampling noise rather than bias.
"""
import os

import policy as base

SAMPLES = int(os.environ.get("ACTION_SAMPLES", "8"))
original_init = base.JointMemoryPolicy.__init__


def patched_init(self, checkpoint_dir):
    original_init(self, checkpoint_dir)
    import jax
    import jax.numpy as jnp

    single = self._sample
    if os.environ.get("ACTION_SEED"):
        # A different flow-noise stream, for repeated closed-loop runs on the same initial states.
        self._rng = jax.random.PRNGKey(int(os.environ["ACTION_SEED"]))

    def averaged(rng, observation, prefix, valid):
        chunks = [single(key, observation, prefix, valid) for key in jax.random.split(rng, SAMPLES)]
        return jnp.mean(jnp.stack(chunks), axis=0)

    self._sample = averaged


base.JointMemoryPolicy.__init__ = patched_init

if __name__ == "__main__":
    base.main()
