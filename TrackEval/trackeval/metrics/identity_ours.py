import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils

class Identity(_BaseMetric):
    """
    Class implementing ID metrics for tracking evaluation.
    """

    @staticmethod
    def get_default_config():
        """
        Return default configuration values.

        Returns:
            dict: Default configuration with the similarity threshold and print option.
        """
        return {
            'THRESHOLD': 0.5,
            'PRINT_CONFIG': True,
            'TRACKLET_CONSISTENCY_THRESHOLD': 0.7  # Default threshold for tracklet consistency
        }

    def __init__(self, config=None):
        super().__init__()

        self.integer_fields = ['IDTP', 'IDFN', 'IDFP']
        self.float_fields = ['IDF1', 'IDR', 'IDP', 'IDP_TrackletConsistency']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.fields

        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])
        self.tracklet_consistency_threshold = float(self.config['TRACKLET_CONSISTENCY_THRESHOLD'])

    @_timing.time
    def eval_sequence(self, data):
        """
        Evaluate ID metrics for a single sequence.

        Args:
            data (dict): Input data containing ground truth and tracker details.

        Returns:
            dict: Calculated metrics for the sequence.
        """
        # Initialize metrics
        res = {field: 0 for field in self.fields}

        # Handle empty sequences
        if data['num_tracker_dets'] == 0:
            res['IDFN'] = data['num_gt_dets']
            return res
        if data['num_gt_dets'] == 0:
            res['IDFP'] = data['num_tracker_dets']
            return res

        # Initialize variables
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros(data['num_gt_ids'])
        tracker_id_count = np.zeros(data['num_tracker_ids'])

        tracker_to_gt_map = {tracker_id: set() for tracker_id in range(data['num_tracker_ids'])}

        # Accumulate global track information with tracklet consistency threshold
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            matches_mask = data['similarity_scores'][t] >= self.threshold
            bbox_matches_mask = data['similarity_scores'][t] >= self.tracklet_consistency_threshold
            combined_mask = matches_mask & bbox_matches_mask

            match_idx_gt, match_idx_tracker = np.nonzero(combined_mask)

            for gt_idx, tracker_idx in zip(match_idx_gt, match_idx_tracker):
                tracker_to_gt_map[tracker_ids_t[tracker_idx]].add(gt_ids_t[gt_idx])

            potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[tracker_ids_t] += 1

        # Prepare cost matrices for Hungarian algorithm
        num_gt_ids, num_tracker_ids = data['num_gt_ids'], data['num_tracker_ids']
        cost_matrix_size = num_gt_ids + num_tracker_ids

        fp_mat = np.zeros((cost_matrix_size, cost_matrix_size))
        fn_mat = np.zeros((cost_matrix_size, cost_matrix_size))

        # Populate cost matrices
        for gt_id in range(num_gt_ids):
            fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
            fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
        for tracker_id in range(num_tracker_ids):
            fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
            fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]

        fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
        fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

        # Perform optimal assignment
        match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)

        # Compute basic statistics
        res['IDFN'] = int(fn_mat[match_rows, match_cols].sum())
        res['IDFP'] = int(fp_mat[match_rows, match_cols].sum())
        res['IDTP'] = int(gt_id_count.sum() - res['IDFN'])

        # Compute tracklet consistency
        tp_tracklet_consistency = 0
        fp_tracklet_consistency = 0

        for tracker_id, gt_ids in tracker_to_gt_map.items():
            if len(gt_ids) == 1:
                tp_tracklet_consistency += 1
            else:
                fp_tracklet_consistency += 1

        try:
            res['IDP_TrackletConsistency'] = (
                tp_tracklet_consistency / (tp_tracklet_consistency + fp_tracklet_consistency)
                if (tp_tracklet_consistency + fp_tracklet_consistency) > 0 else 0
            )
        except:
            res['IDP_TrackletConsistency'] = 0

        # Print tracklet consistency metric
        print(f"AAAAAAA IDP_TrackletConsistency: {res['IDP_TrackletConsistency']}")

        # Compute final ID scores
        return self._compute_final_fields(res)

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """
        Combine metrics across classes by averaging over class values.

        Args:
            all_res (dict): Results from multiple classes.
            ignore_empty_classes (bool): Ignore classes with no detections if True.

        Returns:
            dict: Combined metrics.
        """
        res = {}

        for field in self.integer_fields:
            filtered_res = {k: v for k, v in all_res.items()
                            if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps}
            res[field] = self._combine_sum(filtered_res, field) if ignore_empty_classes else self._combine_sum(all_res, field)

        for field in self.float_fields:
            filtered_values = [v[field] for v in all_res.values()
                               if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps]
            res[field] = np.mean(filtered_values, axis=0) if ignore_empty_classes else np.mean([v[field] for v in all_res.values()], axis=0)

        return res

    def combine_classes_det_averaged(self, all_res):
        """
        Combine metrics across classes by averaging detection values.

        Args:
            all_res (dict): Results from multiple classes.

        Returns:
            dict: Combined metrics.
        """
        res = {field: self._combine_sum(all_res, field) for field in self.integer_fields}
        return self._compute_final_fields(res)

    def combine_sequences(self, all_res):
        """
        Combine metrics across sequences.

        Args:
            all_res (dict): Results from multiple sequences.

        Returns:
            dict: Combined metrics.
        """
        res = {field: self._combine_sum(all_res, field) for field in self.integer_fields}
        return self._compute_final_fields(res)

    @staticmethod
    def _compute_final_fields(res):
        """
        Compute derived metrics based on other metrics.

        Args:
            res (dict): Metrics to compute derived fields from.

        Returns:
            dict: Metrics with derived fields added.
        """
        res['IDR'] = res['IDTP'] / max(1.0, res['IDTP'] + res['IDFN'])
        res['IDP'] = res['IDTP'] / max(1.0, res['IDTP'] + res['IDFP'])
        res['IDF1'] = res['IDTP'] / max(1.0, res['IDTP'] + 0.5 * res['IDFP'] + 0.5 * res['IDFN'])
        return res
