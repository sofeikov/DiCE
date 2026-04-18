"""
Module to generate diverse counterfactual explanations based on PyTorch framework
"""
import copy
import random
import timeit

import numpy as np
import torch
from raiutils.exceptions import UserConfigValidationException

from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase


class DicePyTorch(ExplainerBase):
    CLOSEST_TO_THRESHOLD = "closest_to_threshold"
    MAXIMIZE_DESIRED_CLASS_SCORE = "maximize_desired_class_score"
    COUNTERFACTUAL_SELECTION_STRATEGIES = (
        CLOSEST_TO_THRESHOLD,
        MAXIMIZE_DESIRED_CLASS_SCORE,
    )

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        # initiating data related parameters
        super().__init__(data_interface)
        # initializing model related variables
        self.model = model_interface
        self.model.load_model()  # loading trained model
        self.model.transformer.feed_data_params(data_interface)
        self.model.transformer.initialize_transform_func()
        # temp data to create some attributes like encoded feature names
        temp_ohe_data = self.model.transformer.transform(self.data_interface.data_df.iloc[[0]])
        self.data_interface.create_ohe_params(temp_ohe_data)
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()

        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.ohe_encoded_feature_names)).shape[1]

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate
        self.counterfactual_selection_strategy = self.CLOSEST_TO_THRESHOLD

    def _generate_counterfactuals(self, query_instance, total_CFs,
                                  desired_class="opposite", desired_range=None,
                                  proximity_weight=0.5,
                                  diversity_weight=1.0, categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all",
                                  permitted_range=None, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                  feature_weights="inverse_mad", optimizer="pytorch:adam", learning_rate=0.05, min_iter=500,
                                  max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False,
                                  init_near_query_instance=True, tie_random=False, stopping_threshold=None,
                                  posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear",
                                  limit_steps_ls=10000, best_effort=False,
                                  desired_class_probability_delta=None,
                                  counterfactual_selection_strategy=None):
        """Generates diverse counterfactual explanations.

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class. Provide a class index.
                              "opposite" is supported only for binary classification.
        :param desired_range: Not supported currently.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                               Defaults to the range inferred from training data. If None, uses the parameters initialized in
                               data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as
                                values. Default option is "inverse_mad" where the weight for a continuous feature is the
                                inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set;
                                the weight for a categorical feature is equal to 1 by default.
        :param optimizer: PyTorch optimization algorithm. Currently tested only with "pytorch:adam".
        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence.
                                      Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
                                   Defaults to 0.5 when not provided.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param limit_steps_ls: Defines an upper limit for the linear search step in the posthoc_sparsity_enhancement
        :param best_effort: When True, explicitly keeps and returns the best available optimization result even if
                            it does not satisfy the requested stopping_threshold. Candidates are ranked according to
                            ``counterfactual_selection_strategy``. Returned metadata indicates whether each
                            counterfactual met the threshold or is a best-effort approximation.
        :param desired_class_probability_delta: Optional relative uplift for the desired-class probability/score.
                                                DiCE resolves the effective target threshold per query as the current
                                                desired-class score plus this delta. Classification only; cannot be
                                                combined with ``stopping_threshold``.
        :param counterfactual_selection_strategy: Candidate-ranking strategy for the PyTorch gradient explainer.
                                                  ``closest_to_threshold`` preserves the legacy behavior and
                                                  ``maximize_desired_class_score`` keeps the best desired-class
                                                  probability/score seen during optimization.

        :return: A CounterfactualExamples object to store and visualize the resulting
                 counterfactual explanations (see diverse_counterfactuals.py).
        """
        if not isinstance(best_effort, bool):
            raise UserConfigValidationException("The best_effort parameter should be a boolean.")
        self.counterfactual_selection_strategy = self._resolve_counterfactual_selection_strategy(
            counterfactual_selection_strategy
        )

        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

        # check permitted range for continuous features
        if permitted_range is not None:
            self.data_interface.permitted_range = permitted_range
            self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
            self.cont_minx = []
            self.cont_maxx = []
            for feature in self.data_interface.continuous_feature_names:
                self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

        if [total_CFs, algorithm, features_to_vary] != self.cf_init_weights:
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        if [yloss_type, diversity_loss_type, feature_weights] != self.loss_weights:
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
        if [proximity_weight, diversity_weight, categorical_penalty] != self.hyperparameters:
            self.update_hyperparameters(proximity_weight, diversity_weight, categorical_penalty)

        final_cfs_df, test_instance_df, final_cfs_df_sparse = \
            self.find_counterfactuals(
                query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter,
                project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance,
                tie_random, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm,
                limit_steps_ls, best_effort, desired_class_probability_delta)

        desired_class_param = desired_class
        if self.model.model_type == ModelTypes.Classifier:
            desired_class_param = self.target_cf_class

        return exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class_param,
            metadata=self._build_counterfactual_metadata(best_effort))

    def get_model_output(self, input_instance,
                         transform_data=False, out_tensor=True):
        """get output probability of ML model"""
        return self.model.get_output(
                input_instance,
                transform_data=transform_data,
                out_tensor=out_tensor)

    @staticmethod
    def _flatten_model_output(model_output):
        if hasattr(model_output, "shape") and len(model_output.shape) > 1:
            return model_output[0]
        return model_output

    def _get_class_scores(self, input_instance):
        model_output = self._flatten_model_output(self.get_model_output(input_instance))
        if len(model_output) == 1:
            positive_score = torch.clamp(model_output[0], min=0.0, max=1.0)
            return torch.stack((1 - positive_score, positive_score))
        return model_output

    def predict_fn(self, input_instance):
        """prediction function"""
        if not torch.is_tensor(input_instance):
            input_instance = torch.tensor(input_instance).float()
        model_output = self.get_model_output(
                input_instance, transform_data=False, out_tensor=False)
        return np.asarray(self._flatten_model_output(model_output), dtype=np.float32)

    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        input_instance = self.model.transformer.transform(input_instance).to_numpy(dtype=np.float64)[0]
        return self.predict_fn(torch.tensor(input_instance).float())

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes CFs and other related variables."""

        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1          # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        if features_to_vary != self.features_to_vary:
            self.features_to_vary = features_to_vary
            self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)

        # CF initialization
        if len(self.cfs) != self.total_CFs:
            self.cfs = []
            for ix in range(self.total_CFs):
                one_init = []
                for jx in range(self.minx.shape[1]):
                    one_init.append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
                self.cfs.append(torch.tensor(one_init).float())
                self.cfs[ix].requires_grad = True

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]

        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=True)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1/normalized_mads[feature], 2)

            feature_weights_list = []
            for feature in self.data_interface.ohe_encoded_feature_names:
                if feature in feature_weights:
                    feature_weights_list.append(feature_weights[feature])
                else:
                    feature_weights_list.append(1.0)
            self.feature_weights_list = torch.tensor(feature_weights_list)

        # define different parts of loss function
        self.yloss_opt = torch.nn.BCEWithLogitsLoss()

    def update_hyperparameters(self, proximity_weight, diversity_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.hyperparameters = [proximity_weight, diversity_weight, categorical_penalty]
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty

    def do_optimizer_initializations(self, optimizer, learning_rate):
        """Initializes gradient-based PyTorch optimizers."""
        opt_method = optimizer.split(':')[1]

        # optimizater initialization
        if opt_method == "adam":
            self.optimizer = torch.optim.Adam(self.cfs, lr=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.cfs, lr=learning_rate)

    def compute_yloss(self):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        target_cf_class = self.target_cf_class
        eps = 1e-6
        for i in range(self.total_CFs):
            class_scores = torch.clamp(self._get_class_scores(self.cfs[i]), min=eps, max=1 - eps)
            target_score = class_scores[target_cf_class]
            if self.yloss_type == "l2_loss":
                temp_loss = torch.pow(1 - target_score, 2)
            elif self.yloss_type == "log_loss":
                temp_loss = -torch.log(target_score)
            elif self.yloss_type == "hinge_loss":
                temp_logit = torch.log(target_score / (1 - target_score))
                temp_loss = torch.relu(torch.ones_like(temp_logit) - temp_logit)

            yloss += temp_loss

        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0)

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(self.cfs[i], self.x1)
        return proximity_loss/(torch.mul(len(self.minx[0]), self.total_CFs))

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = torch.ones((self.total_CFs, self.total_CFs))
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i, j)] = 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))
                    if i == j:
                        det_entries[(i, j)] += 0.0001

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i, j)] = 1.0/(torch.exp(self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_entries[(i, j)] += 0.0001

        diversity_loss = torch.det(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(submethod)
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))

            return 1.0 - (diversity_loss/count)

    def compute_regularization_loss(self):
        """Adds a linear equality constraints to the loss functions -
           to ensure all levels of a categorical variable sums to one"""
        regularization_loss = 0.0
        for i in range(self.total_CFs):
            for v in self.encoded_categorical_feature_indexes:
                regularization_loss += torch.pow((torch.sum(self.cfs[i][v[0]:v[-1]+1]) - 1.0), 2)

        return regularization_loss

    def compute_loss(self):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0
        self.regularization_loss = self.compute_regularization_loss()

        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) - \
            (self.diversity_weight * self.diversity_loss) + \
            (self.categorical_penalty * self.regularization_loss)
        return self.loss

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        self.cfs[n].data[i] = query_instance[i]+(n*0.01)
                    else:
                        self.cfs[n].data[i] = np.random.uniform(self.minx[0][i], self.maxx[0][i])
                else:
                    self.cfs[n].data[i] = query_instance[i]

    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = tcf.detach().clone().numpy()
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                # continuous feature in orginal scale
                org_cont = (cf[v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i]
                org_cont = round(org_cont, self.cont_precisions[i])  # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                cf[v] = normalized_cont  # assign the projected continuous value

            for v in self.encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[v[0]:v[-1]+1] == np.amax(cf[v[0]:v[-1]+1])).flatten().tolist()
                if len(maxs) > 1:
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                else:
                    ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix:
                        cf[v[vi]] = 1.0
                    else:
                        cf[v[vi]] = 0.0

            temp_cfs.append(cf)
            if assign:
                for jx in range(len(cf)):
                    self.cfs[index].data[jx] = torch.tensor(temp_cfs[index])[jx]

        if assign:
            return None
        else:
            return temp_cfs

    def stop_loop(self, itr, loss_diff):
        """Determines the stopping condition for gradient descent."""

        # intermediate projections
        if self.project_iter > 0 and itr > 0:
            if itr % self.project_iter == 0:
                self.round_off_cfs(assign=True)

        # do GD for min iterations
        if itr < self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= self.max_iter:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                temp_cfs = self.round_off_cfs(assign=False)
                test_preds = [self.predict_fn(cf) for cf in temp_cfs]

                if all(self.is_cf_valid(pred) for pred in test_preds):
                    self.converged = True
                    return True
                return False
        else:
            self.loss_converge_iter = 0
            return False

    def _get_average_prediction_distance_from_threshold(self, predictions):
        return np.mean(
            [abs(self.get_target_class_score(pred) - self.stopping_threshold) for pred in predictions]
        )

    def _resolve_counterfactual_selection_strategy(self, counterfactual_selection_strategy):
        if counterfactual_selection_strategy is None:
            return self.CLOSEST_TO_THRESHOLD

        if counterfactual_selection_strategy in self.COUNTERFACTUAL_SELECTION_STRATEGIES:
            return counterfactual_selection_strategy

        raise UserConfigValidationException(
            "The counterfactual_selection_strategy should be {0} and not {1}".format(
                " or ".join(self.COUNTERFACTUAL_SELECTION_STRATEGIES),
                counterfactual_selection_strategy,
            )
        )

    def _score_predictions_for_selection(self, predictions):
        if self.counterfactual_selection_strategy == self.CLOSEST_TO_THRESHOLD:
            return float(self._get_average_prediction_distance_from_threshold(predictions))

        return float(np.mean([self.get_target_class_score(pred) for pred in predictions]))

    def _is_better_selection_score(self, selection_score, best_selection_score):
        if best_selection_score is None:
            return True

        if self.counterfactual_selection_strategy == self.CLOSEST_TO_THRESHOLD:
            return selection_score < best_selection_score

        return selection_score > best_selection_score

    @staticmethod
    def _restore_backup_candidates(final_cfs, cfs_preds, total_CFs, loop_ix, backup_cfs, backup_preds):
        for ix in range(total_CFs):
            final_cfs[loop_ix + ix] = copy.deepcopy(backup_cfs[loop_ix + ix])
            cfs_preds[loop_ix + ix] = copy.deepcopy(backup_preds[loop_ix + ix])

    @staticmethod
    def _store_backup_candidates(backup_cfs, backup_preds, start_index, candidates, predictions):
        for ix, candidate in enumerate(candidates):
            backup_cfs[start_index + ix] = copy.deepcopy(candidate)
            backup_preds[start_index + ix] = copy.deepcopy(predictions[ix])

    def _build_counterfactual_metadata(self, best_effort):
        return self.build_counterfactual_metadata(self.cfs_preds, best_effort)

    def _update_backup_candidates(self, loop_ix, candidates, predictions):
        selection_score = self._score_predictions_for_selection(predictions)

        if self._is_better_selection_score(selection_score, self.best_effort_backup_selection_score[loop_ix]):
            self.best_effort_backup_selection_score[loop_ix] = selection_score
            self._store_backup_candidates(
                self.best_effort_backup_cfs,
                self.best_effort_backup_cfs_preds,
                loop_ix,
                candidates,
                predictions,
            )

        if all(self.is_cf_valid(pred) for pred in predictions) and \
                self._is_better_selection_score(selection_score, self.best_backup_selection_score[loop_ix]):
            self.best_backup_selection_score[loop_ix] = selection_score
            self._store_backup_candidates(
                self.best_backup_cfs,
                self.best_backup_cfs_preds,
                loop_ix,
                candidates,
                predictions,
            )

    def _restore_backup_cfs_if_needed(self, loop_find_CFs, best_effort):
        final_cfs_are_valid = not any(not self.is_cf_valid(pred) for pred in self.cfs_preds)
        should_restore_valid_backups = (
            final_cfs_are_valid and
            self.counterfactual_selection_strategy == self.MAXIMIZE_DESIRED_CLASS_SCORE
        )

        if final_cfs_are_valid and not should_restore_valid_backups:
            return

        for loop_ix in range(loop_find_CFs):
            if self.best_backup_selection_score[loop_ix] is not None:
                self._restore_backup_candidates(
                    self.final_cfs,
                    self.cfs_preds,
                    self.total_CFs,
                    loop_ix,
                    self.best_backup_cfs,
                    self.best_backup_cfs_preds,
                )
                continue

            if best_effort and self.best_effort_backup_selection_score[loop_ix] is not None:
                self._restore_backup_candidates(
                    self.final_cfs,
                    self.cfs_preds,
                    self.total_CFs,
                    loop_ix,
                    self.best_effort_backup_cfs,
                    self.best_effort_backup_cfs_preds,
                )

    def _get_return_indices(self, total_counterfactuals, best_effort, minutes, seconds):
        if all(self.is_cf_valid(pred) for pred in self.cfs_preds):
            self.total_CFs_found = total_counterfactuals
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  minutes, 'min %02d' % seconds, 'sec')
            return [ix for ix in range(total_counterfactuals)]

        valid_ix = []
        for cf_ix, pred in enumerate(self.cfs_preds):
            if self.is_cf_valid(pred):
                valid_ix.append(cf_ix)

        self.total_CFs_found = len(valid_ix)
        if self.total_CFs_found == 0:
            if best_effort:
                print('No threshold-satisfying counterfactuals found; returning best-effort counterfactuals.',
                      '; total time taken: %02d' % minutes, 'min %02d' % seconds, 'sec')
            else:
                print('No Counterfactuals found for the given configuation, ',
                      'perhaps try with different values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % minutes, 'min %02d' % seconds, 'sec')
        else:
            if best_effort:
                print('Only %d (required %d)' % (self.total_CFs_found, total_counterfactuals),
                      ' threshold-satisfying counterfactuals found; returning best-effort counterfactuals for the',
                      ' remaining results.', '; total time taken: %02d' % minutes, 'min %02d' % seconds, 'sec')
            else:
                print('Only %d (required %d)' % (self.total_CFs_found, total_counterfactuals),
                      ' Diverse Counterfactuals found for the given configuation, perhaps try with different',
                      ' values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % minutes, 'min %02d' % seconds, 'sec')

        if best_effort:
            return [ix for ix in range(total_counterfactuals)]
        return valid_ix

    def find_counterfactuals(self, query_instance, desired_class, optimizer, learning_rate, min_iter,
                             max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose,
                             init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param,
                             posthoc_sparsity_algorithm, limit_steps_ls, best_effort,
                             desired_class_probability_delta):
        """Finds counterfactuals by gradient-descent."""
        query_instance = self.model.transformer.transform(query_instance).to_numpy(dtype=np.float64)[0]
        self.x1 = torch.tensor(query_instance)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(torch.tensor(query_instance).float())
        self.target_cf_class = self.infer_target_cfs_class(
            desired_class, test_pred, self.num_output_nodes
        )

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = self._resolve_target_class_stopping_threshold(
            stopping_threshold=stopping_threshold,
            desired_class_probability_delta=desired_class_probability_delta,
            test_pred=test_pred,
        )

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = timeit.default_timer()
        self.final_cfs = []

        # looping the find CFs depending on whether its random initialization or not
        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1

        # variables to backup best known CFs so far in the optimization process -
        # if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_backup_cfs_preds = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_backup_selection_score = [None]*loop_find_CFs
        self.best_effort_backup_cfs = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_effort_backup_cfs_preds = [0]*max(self.total_CFs, loop_find_CFs)
        self.best_effort_backup_selection_score = [None]*loop_find_CFs

        for loop_ix in range(loop_find_CFs):
            # CF init
            if self.total_random_inits > 0:
                self.initialize_CFs(query_instance, False)
            else:
                self.initialize_CFs(query_instance, init_near_query_instance)

            # initialize optimizer
            self.do_optimizer_initializations(optimizer, learning_rate)

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0

            while self.stop_loop(iterations, loss_diff) is False:

                # zero all existing gradients
                self.optimizer.zero_grad()
                self.model.model.zero_grad()

                # get loss and backpropogate
                loss_value = self.compute_loss()
                self.loss.backward()

                # freeze features other than feat_to_vary_idxs
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        if jx not in self.feat_to_vary_idxs:
                            self.cfs[ix].grad[jx] = 0.0

                # update the variables
                self.optimizer.step()

                # projection step
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        self.cfs[ix].data[jx] = torch.clamp(self.cfs[ix][jx], min=self.minx[0][jx], max=self.maxx[0][jx])

                if verbose:
                    if (iterations) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

                # backing up CFs if they are valid
                temp_cfs_stored = self.round_off_cfs(assign=False)
                test_preds_stored = [self.predict_fn(cf) for cf in temp_cfs_stored]
                self._update_backup_candidates(loop_ix, temp_cfs_stored, test_preds_stored)

            # rounding off final cfs - not necessary when inter_project=True
            self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                temp = self.cfs[j].detach().clone().numpy()
                self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time

        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        self._restore_backup_cfs_if_needed(loop_find_CFs, best_effort)

        # convert to the expected numpy array format
        query_instance = np.array([query_instance], dtype=np.float32)
        for tix in range(max(loop_find_CFs, self.total_CFs)):
            self.final_cfs[tix] = np.array([self.final_cfs[tix]], dtype=np.float32)
            self.cfs_preds[tix] = np.array([self.cfs_preds[tix]], dtype=np.float32)

            # if self.final_cfs_sparse is not None:
            #     self.final_cfs_sparse[tix] = np.array([self.final_cfs_sparse[tix]], dtype=np.float32)
            #     self.cfs_preds_sparse[tix] = np.array([self.cfs_preds_sparse[tix]], dtype=np.float32)
            #
            if isinstance(self.best_backup_cfs[0], np.ndarray):  # checking if CFs are backed
                self.best_backup_cfs[tix] = np.array([self.best_backup_cfs[tix]], dtype=np.float32)
                self.best_backup_cfs_preds[tix] = np.array([self.best_backup_cfs_preds[tix]], dtype=np.float32)

        # do inverse transform of CFs to original user-fed format
        cfs = np.array([self.final_cfs[i][0] for i in range(len(self.final_cfs))])
        final_cfs_df = self.model.transformer.inverse_transform(
                self.data_interface.get_decoded_data(cfs))
        # rounding off to 3 decimal places
        if self.num_output_nodes == 1:
            cfs_preds = [np.round(np.asarray(preds).reshape(-1)[0], 3) for preds in self.cfs_preds]
            test_pred_value = np.round(np.asarray(test_pred).reshape(-1)[0], 3)
        else:
            cfs_preds = [int(np.argmax(np.asarray(preds).reshape(-1))) for preds in self.cfs_preds]
            test_pred_value = int(np.argmax(np.asarray(test_pred).reshape(-1)))
        final_cfs_df[self.data_interface.outcome_name] = np.array(cfs_preds)

        test_instance_df = self.model.transformer.inverse_transform(
                self.data_interface.get_decoded_data(query_instance))
        test_instance_df[self.data_interface.outcome_name] = np.array([test_pred_value])

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse,
                                                                       test_instance_df,
                                                                       posthoc_sparsity_param,
                                                                       posthoc_sparsity_algorithm,
                                                                       limit_steps_ls)
        else:
            final_cfs_df_sparse = None

        m, s = divmod(self.elapsed, 60)
        returned_ix = self._get_return_indices(max(loop_find_CFs, self.total_CFs), best_effort, m, s)

        if final_cfs_df_sparse is not None:
            final_cfs_df_sparse = final_cfs_df_sparse.iloc[returned_ix].reset_index(drop=True)

        return final_cfs_df.iloc[returned_ix].reset_index(drop=True), test_instance_df, final_cfs_df_sparse
