
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils

class MAYI(_BaseMetric):
    """Class which implements the MAYI metrics"""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a TP match. Default 0.5.
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        main_integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'MT', 'PT', 'ML', 'Frag', 'IDSW', 'IDAS', 'IDSW_R', 'IDTF', 'IDMG', 'IDTF_R', 'IDEX', 'NUM_GT', 'NUM_HYP', 'IDSW_T', 'IDAS_T', 'IDTF_T', 'IDMG_T', 'ID_ERROR', 'TIDTP', 'TIDFP', 'TIDTN', 'TIDFN']
        extra_integer_fields = ['CLR_Frames']
        self.integer_fields = main_integer_fields + extra_integer_fields
        main_float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA', 'TIDP', 'TIDR']
        extra_float_fields = ['CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']
        self.float_fields = main_float_fields + extra_float_fields
        self.fields = self.float_fields + self.integer_fields
        self.summed_fields = self.integer_fields + ['MOTP_sum']
        self.summary_fields = ['MOTA', 'Frag', 'ID_ERROR', 'IDSW', 'IDAS', 'IDSW_R', 'IDMG', 'IDTF_R', 'IDEX', 'NUM_GT', 'NUM_HYP', 'TIDP', 'TIDR']

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])


    @_timing.time
    def eval_sequence(self, data):
        # print(data.keys())
        # input()
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['CLR_FN'] = data['num_gt_dets']
            res['ML'] = data['num_gt_ids']
            res['MLR'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['CLR_FP'] = data['num_tracker_dets']
            res['MLR'] = 1.0
            return res

        # Variables counting global association
        num_gt_ids = data['num_gt_ids']
        num_tracker_ids = data['num_tracker_ids']

        res['NUM_GT'] = num_gt_ids
        res['NUM_HYP'] = num_tracker_ids

        gt_id_count = np.zeros(num_gt_ids)  # For MT/ML/PT
        gt_matched_count = np.zeros(num_gt_ids)  # For MT/ML/PT
        gt_frag_count = np.zeros(num_gt_ids)  # For Frag

        # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
        # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
        prev_tracker_id = np.nan * np.zeros(num_gt_ids)  # For scoring IDSW
        prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)  # For matching IDSW

        prev_gt_id = np.nan * np.zeros(num_tracker_ids)  # For scoring IDSW
        prev_timestep_gt_id = np.nan * np.zeros(num_tracker_ids)  # For matching IDSW

        existing_gt_ids = np.zeros(num_gt_ids, dtype=bool)
        existing_tracker_ids = np.zeros(num_tracker_ids, dtype=bool)

        prev_history_ts_gt_ids = np.zeros(num_gt_ids)
        prev_history_ts_tracker_ids = np.zeros(num_tracker_ids)

        last_founded_ts_gt_ids = np.zeros(num_gt_ids)
        last_founded_ts_tracker_ids = np.zeros(num_tracker_ids)



        tracker_map = {}
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            similarity = data['similarity_scores'][t]
            similarity[similarity < 0.5] = 0
            gt_indices, tracker_indices = linear_sum_assignment(-similarity)

            for i, (mat_gt_id, mat_tracker_id) in enumerate(zip(gt_indices, tracker_indices)):                
                gt_id = gt_ids_t[mat_gt_id]  # real ID
                tracker_id = tracker_ids_t[mat_tracker_id] # real ID
                tracker_map.setdefault(tracker_id, {}).setdefault(t, {})['matched_gt_id'] = gt_id
                tracker_map.setdefault(tracker_id, {}).setdefault(t, {})['det'] = data['tracker_dets'][t][i]

        # 모든 GT ID와 Tracker ID를 추적
        all_gt_ids = set()
        all_tracker_ids = set()
        for t in range(len(data['gt_ids'])):
            all_gt_ids.update(data['gt_ids'][t])
            all_tracker_ids.update(data['tracker_ids'][t])

        # 매칭된 GT ID와 Tracker ID 추적
        matched_gt_ids = set()
        matched_tracker_ids = set()
        for tid, tid_data in tracker_map.items():
            for fid, fid_data in tid_data.items():
                matched_gt_ids.add(fid_data['matched_gt_id'])
                matched_tracker_ids.add(tid)

        # count TP, FP
        res['TIDTP'] = 0
        res['TIDFP'] = 0
        for tid, tid_data in tracker_map.items():
            matched_gt_ids =  set()
            for fid, fid_data in tid_data.items():
                matched_gt_ids.add(fid_data['matched_gt_id'])
            if len(matched_gt_ids) == 1:
                res['TIDTP'] += 1
            else:
                res['TIDFP'] += 1

        # TN, FN 계산
        res['TIDTN'] = len(all_gt_ids - matched_gt_ids)  # 매칭되지 않은 GT ID는 TN
        res['TIDFN'] = len(all_tracker_ids - matched_tracker_ids)  # 매칭되지 않은 Tracker ID는 FN



        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                res['CLR_FP'] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                res['CLR_FN'] += len(gt_ids_t)
                gt_id_count[gt_ids_t] += 1
                continue

            # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily
            similarity = data['similarity_scores'][t]
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < self.threshold - np.finfo('float').eps] = 0

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]

            # Calc IDSW for MOTA
            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            prev_matched_gt_ids = prev_gt_id[matched_tracker_ids]


            is_tracker_id_changed = np.not_equal(matched_tracker_ids, prev_matched_tracker_ids)
            is_gt_id_changed = np.not_equal(matched_gt_ids, prev_matched_gt_ids)


            is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & ( # GT가 새로 생성된 것이 아님
                np.not_equal(matched_tracker_ids, prev_matched_tracker_ids)) # matched hypothesis와 prev hypothesis가 다름
            res['IDSW'] += np.sum(is_idsw)

            
            idsw_duration = t - prev_history_ts_gt_ids[matched_gt_ids[is_idsw]] # IDSW 일어나기 이전까지 얼마나의 time이 있었냐
            res['IDSW_T'] += np.sum(idsw_duration > 30)

            is_idas = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & ( # GT가 새로 생성된 것이 아님
                np.not_equal(matched_tracker_ids, prev_matched_tracker_ids)) & ( # matched hypothesis와 prev hypothesis가 다름
                np.logical_not(existing_tracker_ids[matched_tracker_ids])) # hypothesis가 기존에 존재하지 않았어야 함
            res['IDAS'] += np.sum(is_idas)

            idas_duration = t - prev_history_ts_gt_ids[matched_gt_ids[is_idas]] # IDSW 일어나기 이전까지 얼마나의 time이 있었냐
            res['IDAS_T'] += np.sum(idas_duration > 30)

            is_idtf = (np.logical_not(np.isnan(prev_matched_gt_ids))) & ( # Hypothesis가 새로 생성된 것이 아님
                np.not_equal(matched_gt_ids, prev_matched_gt_ids)) # matched gt와 prev gt가 다름
            res['IDTF'] += np.sum(is_idtf)

            idtf_duration = t - prev_history_ts_tracker_ids[matched_tracker_ids[is_idtf]] # IDTF 일어나기 이전까지 얼마나의 time이 있었냐
            res['IDTF_T'] += np.sum(idtf_duration > 30)

            is_idmg = (np.logical_not(np.isnan(prev_matched_gt_ids))) & ( # Hypothesis가 새로 생성된 것이 아님
                np.not_equal(matched_gt_ids, prev_matched_gt_ids)) & ( # matched gt와 prev gt가 다름
                np.logical_not(existing_gt_ids[matched_gt_ids])) # gt가 기존에 존재하지 않았어야 함
            res['IDMG'] += np.sum(is_idmg)
            
            idmg_duration = t - prev_history_ts_tracker_ids[matched_tracker_ids[is_idmg]] # IDTF 일어나기 이전까지 얼마나의 time이 있었냐
            res['IDMG_T'] += np.sum(idmg_duration > 30)

            is_idsw_r = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & ( # GT가 새로 생성된 것이 아님
            np.not_equal(matched_tracker_ids, prev_matched_tracker_ids)) & ( # matched hypothesis와 prev hypothesis가 다름
            np.equal(matched_gt_ids, prev_matched_gt_ids)) & ( # hypothesis가 기존에 존재하지 않았어야 함
            existing_tracker_ids[matched_tracker_ids])
            res['IDSW_R'] += np.sum(is_idsw_r)

            is_idtf_r = (np.logical_not(np.isnan(prev_matched_gt_ids))) & ( # Hypothesis가 새로 생성된 것이 아님
            np.not_equal(matched_gt_ids, prev_matched_gt_ids)) & ( # matched gt와 prev gt가 다름
            np.equal(matched_tracker_ids, prev_matched_tracker_ids)) & ( # gt가 기존에 존재하지 않았어야 함
            existing_gt_ids[matched_gt_ids])
            res['IDTF_R'] += np.sum(is_idtf_r)

            res['IDEX'] = res['IDTF'] - res['IDMG'] - res['IDTF_R']
            res['ID_ERROR'] = res['IDAS'] + res['IDSW_R'] + res['IDMG'] + res['IDTF_R'] + res['IDEX']


            # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestep
            gt_id_count[gt_ids_t] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)

            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

            prev_gt_id[matched_tracker_ids] = matched_gt_ids
            prev_timestep_gt_id[:] = np.nan
            prev_timestep_gt_id[matched_tracker_ids] = matched_gt_ids

            currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)


            # temp_gt_idx = np.logical_not(np.isnan(prev_timestep_gt_id[prev_matched_tracker_ids[np.logical_not(np.isnan(prev_matched_tracker_ids))].astype(int)]))
            # temp_matched_gt_ids = np.zeros(len(match_rows), dtype=bool)
            # temp_matched_gt_ids[np.logical_not(np.isnan(prev_matched_tracker_ids))] = temp_gt_idx

            # temp_tracker_idx = np.logical_not(np.isnan(prev_timestep_tracker_id[prev_matched_gt_ids[np.logical_not(np.isnan(prev_matched_gt_ids))].astype(int)]))
            # temp_matched_tracker_ids = np.zeros(len(match_cols), dtype=bool)
            # temp_matched_tracker_ids[np.logical_not(np.isnan(prev_matched_gt_ids))] = temp_tracker_idx

            # is_idsw_r = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & ( # GT가 새로 생성된 것이 아님
            # np.not_equal(matched_tracker_ids, prev_matched_tracker_ids)) & ( # matched hypothesis와 prev hypothesis가 다름
            # temp_matched_gt_ids) & ( # hypothesis가 기존에 존재하지 않았어야 함
            # existing_tracker_ids[matched_tracker_ids])
            # res['IDSW_R'] += np.sum(is_idsw_r)

            # is_idtf_r = (np.logical_not(np.isnan(prev_matched_gt_ids))) & ( # Hypothesis가 새로 생성된 것이 아님
            # np.not_equal(matched_gt_ids, prev_matched_gt_ids)) & ( # matched gt와 prev gt가 다름
            # temp_matched_tracker_ids) & ( # gt가 기존에 존재하지 않았어야 함
            # existing_gt_ids[matched_gt_ids])
            # res['IDTF_R'] += np.sum(is_idtf_r)


            existing_gt_ids[matched_gt_ids] = True
            existing_tracker_ids[matched_tracker_ids] = True

            prev_history_ts_gt_ids[matched_gt_ids[is_gt_id_changed]] = t
            prev_history_ts_tracker_ids[matched_tracker_ids[is_tracker_id_changed]] = t

            last_founded_ts_gt_ids[matched_gt_ids] = t
            last_founded_ts_tracker_ids[matched_tracker_ids] = t

            # Calculate and accumulate basic statistics
            num_matches = len(matched_gt_ids)
            res['CLR_TP'] += num_matches
            res['CLR_FN'] += len(gt_ids_t) - num_matches
            res['CLR_FP'] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                res['MOTP_sum'] += sum(similarity[match_rows, match_cols])

        # Calculate MT/ML/PT/Frag/MOTP
        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res['MT'] = np.sum(np.greater(tracked_ratio, 0.8))
        res['PT'] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - res['MT']
        res['ML'] = num_gt_ids - res['MT'] - res['PT']
        res['Frag'] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1))
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])

        res['CLR_Frames'] = data['num_timesteps']

        # Calculate final CLEAR scores
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0}, field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                res[field] = np.mean(
                    [v[field] for v in all_res.values() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0], axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        num_gt_ids = res['MT'] + res['ML'] + res['PT']
        res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
        res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
        res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
        res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FP'])
        res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])

        res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5*res['CLR_FN'] + 0.5*res['CLR_FP'])
        res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
        safe_log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
        res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['TIDP'] = res['TIDTP'] / (res['TIDTP'] + res['TIDFP'] + 1e-10)
        res['TIDR'] = res['TIDTP'] / (res['TIDTP'] + res['TIDFN'] + 1e-10)
        return res