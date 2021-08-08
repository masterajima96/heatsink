
import ray
from ray.rllib.optimizers import SyncReplayOptimizer
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.compression import pack_if_needed
from ray.rllib.utils.memory import ray_get_and_free


class CustomSyncReplayOptimizer(SyncReplayOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        with self.update_weights_timer:
            if self.remote_evaluators:
                weights = ray.put(self.local_evaluator.get_weights())
                for e in self.remote_evaluators:
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.remote_evaluators:
                batch = SampleBatch.concat_samples(
                    ray_get_and_free(
                        [e.sample.remote() for e in self.remote_evaluators]))
            else:
                batch = self.local_evaluator.sample()

            # Handle everything as if multiagent
            if isinstance(batch, SampleBatch):
                batch = MultiAgentBatch({
                    DEFAULT_POLICY_ID: batch
                }, batch.count)

            for policy_id, s in batch.policy_batches.items():
                for row in s.rows():
                    self.replay_buffers[policy_id].add(
                        pack_if_needed(row["obs"]),
                        row["actions"],
                        row["rewards"],
                        pack_if_needed(row["new_obs"]),
                        row["dones"],
                        weight=None)

        if self.num_steps_sampled >= self.replay_starts:
            self._optimize()

        self.num_steps_sampled += batch.count
        return batch