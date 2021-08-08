from ray.tune.logger import *
import tensorflow as tf


class HouseLogger(TFLogger):

    def on_result(self, result):
        tmp = result.copy()
        if not tmp.get('extra_metrics'):
            tmp['extra_metrics'] = {}
        for metric, values in tmp['extra_metrics'].items():
            if 'temperature' in metric:
                values = np.array(values.popitem()[1])
                if values.size:
                    tmp[f'{metric}_max'] = values.max()
                    tmp[f'{metric}_min'] = values.min()
                    tmp[f'{metric}_mean'] = np.mean(values)
                    tmp[f'{metric}_sd'] = np.std(values)
            else:
                tmp[metric] = sum(values.popitem()[1])
        for k in [
            "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]  # not useful to tf log these
        values = to_tf_values(tmp, ["ray", "tune"])
        train_stats = tf.Summary(value=values)
        t = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self._file_writer.add_summary(train_stats, t)
        iteration_value = to_tf_values({
            "training_iteration": result[TRAINING_ITERATION]
        }, ["ray", "tune"])
        iteration_stats = tf.Summary(value=iteration_value)
        self._file_writer.add_summary(iteration_stats, t)
        self._file_writer.flush()