import re
from types import MethodType

import numpy as np
import pandas as pd
import pytest
from rai_test_utils.datasets.tabular import create_housing_data
from raiutils.exceptions import UserConfigValidationException
from sklearn.ensemble import RandomForestRegressor

import dice_ml
from dice_ml.diverse_counterfactuals import CounterfactualExamples
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml.utils import helpers

from ..conftest import _load_custom_testing_binary_model


@pytest.mark.parametrize("method", ['random', 'genetic', 'kdtree'])
class TestExplainerBaseBinaryClassification:

    def _verify_feature_importance(self, feature_importance):
        if feature_importance is not None:
            for key in feature_importance:
                assert feature_importance[key] >= 0.0
                assert feature_importance[key] <= 1.0

    def test_check_any_counterfactuals_computed(
        self, method,
        custom_public_data_interface,
        sklearn_binary_classification_model_interface
    ):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)

        sample_custom_query = custom_public_data_interface.data_df[0:1]
        cf_example = CounterfactualExamples(
            data_interface=custom_public_data_interface,
            test_instance_df=sample_custom_query)
        cf_examples_arr = [cf_example]

        with pytest.raises(
                UserConfigValidationException,
                match="No counterfactuals found for any of the query points! Kindly check your configuration."):
            exp._check_any_counterfactuals_computed(cf_examples_arr=cf_examples_arr)

        cf_example_has_cf = CounterfactualExamples(
            data_interface=custom_public_data_interface,
            final_cfs_df=sample_custom_query,
            test_instance_df=sample_custom_query)
        cf_example_no_cf = CounterfactualExamples(
            data_interface=custom_public_data_interface,
            test_instance_df=sample_custom_query)
        cf_examples_arr = [cf_example_has_cf, cf_example_no_cf]
        exp._check_any_counterfactuals_computed(cf_examples_arr=cf_examples_arr)

    @pytest.mark.parametrize("desired_class", [1])
    def test_local_feature_importance(
            self, desired_class, method,
            sample_custom_query_1, sample_counterfactual_example_dummy,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        sample_custom_query = pd.concat([sample_custom_query_1, sample_custom_query_1])
        cf_explanations = exp.generate_counterfactuals(
                    query_instances=sample_custom_query,
                    total_CFs=15,
                    desired_class=desired_class)

        cf_explanations.cf_examples_list[0].final_cfs_df = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[0].final_cfs_df_sparse = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[0].final_cfs_df.drop([0, 1, 2], inplace=True)
        cf_explanations.cf_examples_list[0].final_cfs_df_sparse.drop([0, 1, 2], inplace=True)

        cf_explanations.cf_examples_list[1].final_cfs_df = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[1].final_cfs_df_sparse = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[1].final_cfs_df.drop([0], inplace=True)
        cf_explanations.cf_examples_list[1].final_cfs_df_sparse.drop([0], inplace=True)

        local_importances = exp.local_feature_importance(
            query_instances=None,
            cf_examples_list=cf_explanations.cf_examples_list)

        for local_importance in local_importances.local_importance:
            self._verify_feature_importance(local_importance)

    @pytest.mark.parametrize("desired_class", [1])
    def test_global_feature_importance(
            self, desired_class, method,
            sample_custom_query_10, sample_counterfactual_example_dummy,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)

        cf_explanations = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_10,
                    total_CFs=15,
                    desired_class=desired_class)

        cf_explanations.cf_examples_list[0].final_cfs_df = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[0].final_cfs_df_sparse = sample_counterfactual_example_dummy.copy()
        cf_explanations.cf_examples_list[0].final_cfs_df.drop([0, 1, 2, 3, 4], inplace=True)
        cf_explanations.cf_examples_list[0].final_cfs_df_sparse.drop([0, 1, 2, 3, 4], inplace=True)

        for index in range(1, len(cf_explanations.cf_examples_list)):
            cf_explanations.cf_examples_list[index].final_cfs_df = sample_counterfactual_example_dummy.copy()
            cf_explanations.cf_examples_list[index].final_cfs_df_sparse = sample_counterfactual_example_dummy.copy()

        global_importance = exp.global_feature_importance(
            query_instances=None,
            cf_examples_list=cf_explanations.cf_examples_list)

        self._verify_feature_importance(global_importance.summary_importance)

    @pytest.mark.parametrize("desired_class", [1])
    def test_columns_out_of_order(self, desired_class, method, sample_custom_query_1):
        if method == 'genetic':
            pytest.skip('DiceGenetic explainer fails this test case')

        dataset = helpers.load_outcome_not_last_column_dataset()
        d = dice_ml.Data(
            dataframe=dataset, continuous_features=['Numerical'],
            outcome_name='Outcome')
        model = _load_custom_testing_binary_model()
        m = dice_ml.Model(model=model, backend='sklearn')
        exp = dice_ml.Dice(d, m, method=method)

        exp._generate_counterfactuals(
            query_instance=sample_custom_query_1,
            total_CFs=0,
            desired_class=desired_class,
            desired_range=None,
            permitted_range=None,
            features_to_vary='all')

    @pytest.mark.parametrize("desired_class", [1])
    def test_incorrect_features_to_vary_list(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(
                UserConfigValidationException,
                match="Got features {" + "'unknown_feature'" + "} which are not present in training data"):
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_1,
                total_CFs=10,
                desired_class=desired_class,
                desired_range=None,
                permitted_range=None,
                features_to_vary=['unknown_feature'])

    @pytest.mark.parametrize("desired_class", [1])
    def test_incorrect_features_permitted_range(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(
                UserConfigValidationException,
                match="Got features {" + "'unknown_feature'" + "} which are not present in training data"):
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_1,
                total_CFs=10,
                desired_class=desired_class,
                desired_range=None,
                permitted_range={'unknown_feature': [1, 30]},
                features_to_vary='all')

    @pytest.mark.parametrize("desired_class", [1])
    def test_incorrect_values_permitted_range(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(UserConfigValidationException) as ucve:
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_1,
                total_CFs=10,
                desired_class=desired_class,
                desired_range=None,
                permitted_range={'Categorical': ['d']},
                features_to_vary='all')

        assert 'The category {0} does not occur in the training data for feature {1}. Allowed categories are {2}'.format(
            'd', 'Categorical', ['a', 'b', 'c']) in str(ucve)

    # When no elements in the desired_class are present in the training data
    @pytest.mark.parametrize("desired_class", [100, 'a'])
    def test_unsupported_binary_class(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(UserConfigValidationException) as ucve:
            exp.generate_counterfactuals(query_instances=sample_custom_query_1, total_CFs=3,
                                         desired_class=desired_class)
        if desired_class == 100:
            assert "Desired class not present in training data!" in str(ucve)
        else:
            assert "The target class for {0} could not be identified".format(desired_class) in str(ucve)

    # Testing if an error is thrown when the query instance has an unknown categorical variable
    @pytest.mark.parametrize("desired_class", [1])
    def test_query_instance_unknown_column(
            self, desired_class, method, sample_custom_query_5,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(ValueError, match='not present in training data'):
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_5, total_CFs=3,
                desired_class=desired_class)

    # Testing if an error is thrown when the query instance has an unknown categorical variable
    @pytest.mark.parametrize("desired_class", [1])
    def test_query_instance_outside_bounds(
            self, desired_class, method, sample_custom_query_3,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        with pytest.raises(ValueError, match='has a value outside the dataset'):
            exp.generate_counterfactuals(query_instances=sample_custom_query_3, total_CFs=1,
                                         desired_class=desired_class)

    # # Testing that the counterfactuals are in the desired class
    @pytest.mark.parametrize("desired_class", [1])
    def test_desired_class(
            self, desired_class, method, sample_custom_query_2,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        ans = exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                           features_to_vary='all',
                                           total_CFs=2, desired_class=desired_class,
                                           permitted_range=None)
        if method != 'kdtree':
            assert all(ans.cf_examples_list[0].final_cfs_df[exp.data_interface.outcome_name].values == [desired_class] * 2)
        else:
            assert all(ans.cf_examples_list[0].final_cfs_df_sparse[exp.data_interface.outcome_name].values ==
                       [desired_class] * 2)

        exp.serialize_explainer(method + '.pkl')
        new_exp = ExplainerBase.deserialize_explainer(method + '.pkl')

        ans = new_exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                               features_to_vary='all',
                                               total_CFs=2, desired_class=desired_class,
                                               permitted_range=None)
        if method != 'kdtree':
            assert all(ans.cf_examples_list[0].final_cfs_df[new_exp.data_interface.outcome_name].values == [desired_class] * 2)
        else:
            assert all(ans.cf_examples_list[0].final_cfs_df_sparse[new_exp.data_interface.outcome_name].values ==
                       [desired_class] * 2)

    @pytest.mark.parametrize(
        ("desired_class", "stopping_threshold", "model_score"),
        [(1, 0.4, np.array([0.57, 0.43])), (0, 0.6, np.array([0.6, 0.4]))],
    )
    def test_shared_threshold_logic_uses_target_class_score(
            self, desired_class, stopping_threshold, model_score, method,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        exp.num_output_nodes = 2

        exp.misc_init(
            stopping_threshold=stopping_threshold,
            desired_class=desired_class,
            desired_range=None,
            test_pred=np.array([0.5, 0.5]),
        )

        assert exp.stopping_threshold == pytest.approx(stopping_threshold)
        assert exp.is_cf_valid(model_score) is True

    @pytest.mark.parametrize(
        ("desired_class", "desired_class_probability_delta", "test_pred", "model_score", "expected_threshold"),
        [
            (1, 0.07, np.array([0.64, 0.36]), np.array([0.57, 0.43]), 0.43),
            (0, 0.2, np.array([0.6, 0.4]), np.array([0.8, 0.2]), 0.8),
        ],
    )
    def test_shared_threshold_logic_resolves_desired_class_probability_delta(
            self, desired_class, desired_class_probability_delta, test_pred, model_score, expected_threshold, method,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        exp.num_output_nodes = 2

        exp.misc_init(
            stopping_threshold=None,
            desired_class=desired_class,
            desired_range=None,
            test_pred=test_pred,
            desired_class_probability_delta=desired_class_probability_delta,
        )

        assert exp.stopping_threshold == pytest.approx(expected_threshold)
        assert exp.is_cf_valid(model_score)

    def test_shared_threshold_logic_caps_desired_class_probability_delta_at_one(
            self, method, custom_public_data_interface, sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        exp.num_output_nodes = 2

        with pytest.warns(
                UserWarning,
                match=r'Capping the resolved stopping_threshold at 1.0'):
            exp.misc_init(
                stopping_threshold=None,
                desired_class=1,
                desired_range=None,
                test_pred=np.array([0.02, 0.98]),
                desired_class_probability_delta=0.07,
            )

        assert exp.stopping_threshold == pytest.approx(1.0)

    def test_generate_counterfactuals_preserves_legacy_explainer_signature(
            self, method, sample_custom_query_1, custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method='random')
        captured = {}

        def legacy_generate_counterfactuals(
                self, query_instance, total_CFs, desired_class="opposite", desired_range=None,
                permitted_range=None, features_to_vary="all", stopping_threshold=0.5,
                posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", verbose=False):
            captured["stopping_threshold"] = stopping_threshold
            test_instance_df = query_instance.copy()
            test_instance_df[self.data_interface.outcome_name] = [1]
            final_cfs_df = test_instance_df.copy()
            return CounterfactualExamples(
                data_interface=self.data_interface,
                test_instance_df=test_instance_df,
                final_cfs_df=final_cfs_df,
                final_cfs_df_sparse=final_cfs_df,
                desired_class=1,
            )

        exp._generate_counterfactuals = MethodType(legacy_generate_counterfactuals, exp)

        counterfactual_explanations = exp.generate_counterfactuals(
            query_instances=sample_custom_query_1,
            total_CFs=1,
            desired_class=1,
        )

        assert counterfactual_explanations.cf_examples_list[0].final_cfs_df is not None
        assert captured["stopping_threshold"] == pytest.approx(0.5)

    def test_generate_counterfactuals_rejects_delta_for_legacy_explainer(
            self, method, sample_custom_query_1, custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method='random')

        def legacy_generate_counterfactuals(
                self, query_instance, total_CFs, desired_class="opposite", desired_range=None,
                permitted_range=None, features_to_vary="all", stopping_threshold=0.5,
                posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", verbose=False):
            raise AssertionError("legacy explainer should not be invoked when delta is unsupported")

        exp._generate_counterfactuals = MethodType(legacy_generate_counterfactuals, exp)

        with pytest.raises(
                UserConfigValidationException,
                match=r'The desired_class_probability_delta parameter is not supported by this explainer implementation.'):
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_1,
                total_CFs=1,
                desired_class=1,
                desired_class_probability_delta=0.07,
            )

    def test_generate_counterfactuals_rejects_permitted_direction_for_legacy_explainer(
            self, method, sample_custom_query_1, custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method='random')

        def legacy_generate_counterfactuals(
                self, query_instance, total_CFs, desired_class="opposite", desired_range=None,
                permitted_range=None, features_to_vary="all", stopping_threshold=0.5,
                posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", verbose=False):
            raise AssertionError("legacy explainer should not be invoked when permitted_direction is unsupported")

        exp._generate_counterfactuals = MethodType(legacy_generate_counterfactuals, exp)

        with pytest.raises(
                UserConfigValidationException,
                match=r'The permitted_direction parameter is not supported by this explainer implementation.'):
            exp.generate_counterfactuals(
                query_instances=sample_custom_query_1,
                total_CFs=1,
                desired_class=1,
                permitted_direction={'Numerical': 'increase'},
            )

    @pytest.mark.parametrize(("desired_class", "total_CFs", "permitted_range"),
                             [(1, 1, {'Numerical': [10, 150]})])
    def test_permitted_range(
            self, desired_class, method, total_CFs, permitted_range, sample_custom_query_2,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        ans = exp.generate_counterfactuals(query_instances=sample_custom_query_2,
                                           permitted_range=permitted_range,
                                           total_CFs=total_CFs, desired_class=desired_class)

        for feature in permitted_range:
            if method != 'kdtree':
                assert all(
                    permitted_range[feature][0] <= ans.cf_examples_list[0].final_cfs_df[feature].values[i] <=
                    permitted_range[feature][1] for i in range(total_CFs))
            else:
                assert all(
                    permitted_range[feature][0] <= ans.cf_examples_list[0].final_cfs_df_sparse[feature].values[i] <=
                    permitted_range[feature][1] for i in range(total_CFs))

    @pytest.mark.parametrize(
        ('query_instance', 'desired_class', 'permitted_direction', 'expected_comparison'),
        [
            (pd.DataFrame({'Categorical': ['c'], 'Numerical': [8]}), 0, {'Numerical': 'increase'}, 'ge'),
            (pd.DataFrame({'Categorical': ['c'], 'Numerical': [4]}), 0, {'Numerical': 'decrease'}, 'le'),
        ],
    )
    def test_permitted_direction(
            self, method, query_instance, desired_class, permitted_direction, expected_comparison,
            sklearn_binary_classification_model_interface):
        binary_public_data_interface = dice_ml.Data(
            dataframe=helpers.load_custom_testing_dataset_binary(),
            continuous_features=['Numerical'],
            outcome_name='Outcome',
        )
        exp = dice_ml.Dice(
            binary_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)

        kwargs = {'posthoc_sparsity_param': 0}
        if method == 'random':
            kwargs.update({'sample_size': 2000, 'random_seed': 0})
        elif method == 'genetic':
            kwargs.update({'initialization': 'kdtree', 'maxiterations': 30})

        ans = exp.generate_counterfactuals(
            query_instances=query_instance,
            features_to_vary=['Numerical'],
            permitted_direction=permitted_direction,
            total_CFs=1,
            desired_class=desired_class,
            **kwargs
        )

        cf_examples = ans.cf_examples_list[0]
        final_cfs_df = cf_examples.final_cfs_df_sparse if cf_examples.final_cfs_df_sparse is not None \
            else cf_examples.final_cfs_df

        assert final_cfs_df is not None
        if expected_comparison == 'ge':
            assert final_cfs_df['Numerical'].iloc[0] >= query_instance['Numerical'].iloc[0]
        else:
            assert final_cfs_df['Numerical'].iloc[0] <= query_instance['Numerical'].iloc[0]

    def test_permitted_direction_rejects_empty_intersection(
            self, method, sklearn_binary_classification_model_interface):
        binary_public_data_interface = dice_ml.Data(
            dataframe=helpers.load_custom_testing_dataset_binary(),
            continuous_features=['Numerical'],
            outcome_name='Outcome',
        )
        exp = dice_ml.Dice(
            binary_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        query_instance = pd.DataFrame({'Categorical': ['c'], 'Numerical': [4]})

        kwargs = {'posthoc_sparsity_param': 0}
        if method == 'random':
            kwargs.update({'sample_size': 50, 'random_seed': 0})
        elif method == 'genetic':
            kwargs.update({'initialization': 'kdtree', 'maxiterations': 5})

        with pytest.raises(
                UserConfigValidationException,
                match=r"No valid values remain for feature Numerical after applying permitted_direction='decrease'."):
            exp.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=1,
                desired_class=0,
                permitted_range={'Numerical': [5, 10]},
                permitted_direction={'Numerical': 'decrease'},
                **kwargs
            )

    def test_fixed_continuous_feature_must_stay_within_constraints(
            self, method, sklearn_binary_classification_model_interface):
        binary_public_data_interface = dice_ml.Data(
            dataframe=helpers.load_custom_testing_dataset_binary(),
            continuous_features=['Numerical'],
            outcome_name='Outcome',
        )
        exp = dice_ml.Dice(
            binary_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        query_instance = pd.DataFrame({'Categorical': ['c'], 'Numerical': [8]})

        kwargs = {'posthoc_sparsity_param': 0}
        if method == 'random':
            kwargs.update({'sample_size': 50, 'random_seed': 0})
        elif method == 'genetic':
            kwargs.update({'initialization': 'kdtree', 'maxiterations': 5})

        with pytest.raises(
                ValueError, match="is outside the permitted range and isn't allowed to vary"):
            exp.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=1,
                desired_class=0,
                features_to_vary=['Categorical'],
                permitted_range={'Numerical': [1, 4]},
                **kwargs
            )

    # Testing for 0 CFs needed
    @pytest.mark.parametrize(("features_to_vary", "desired_class", "desired_range", "total_CFs", "permitted_range"),
                             [("all", 0, None, 0, None)])
    def test_zero_cfs_internal(
            self, method, features_to_vary, desired_class, desired_range, sample_custom_query_2, total_CFs,
            permitted_range, custom_public_data_interface, sklearn_binary_classification_model_interface):
        if method == 'genetic':
            pytest.skip('DiceGenetic explainer does not handle the total counterfactuals as zero')
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        features_to_vary = exp.setup(features_to_vary, None, sample_custom_query_2, "inverse_mad")
        exp._generate_counterfactuals(features_to_vary=features_to_vary, query_instance=sample_custom_query_2,
                                      total_CFs=total_CFs, desired_class=desired_class,
                                      desired_range=desired_range, permitted_range=permitted_range)

    @pytest.mark.parametrize("desired_class", [1])
    def test_cfs_type_consistency(
            self, desired_class, method,
            sample_custom_query_1, sample_counterfactual_example_dummy,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)
        sample_custom_query = pd.concat([sample_custom_query_1, sample_custom_query_1])
        cf_explanations = exp.generate_counterfactuals(
                    query_instances=sample_custom_query,
                    total_CFs=2,
                    desired_class=desired_class)
        for col in sample_custom_query.columns:
            assert cf_explanations.cf_examples_list[0].test_instance_df[col].dtype == sample_custom_query[col].dtype
            if cf_explanations.cf_examples_list[0].final_cfs_df is not None:
                assert cf_explanations.cf_examples_list[0].final_cfs_df[col].dtype == sample_custom_query[col].dtype
            if cf_explanations.cf_examples_list[0].final_cfs_df_sparse is not None:
                assert cf_explanations.cf_examples_list[0].final_cfs_df_sparse[col].dtype == sample_custom_query[col].dtype


@pytest.mark.parametrize("method", ['random', 'genetic', 'kdtree'])
class TestExplainerBaseMultiClassClassification:

    @pytest.mark.parametrize("desired_class", [1])
    def test_zero_totalcfs(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface,
            method=method)
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_class=desired_class)

    # Testing that the counterfactuals are in the desired class
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(2, 2)])
    @pytest.mark.parametrize("genetic_initialization", ['kdtree', 'random'])
    def test_desired_class(
            self, desired_class, total_CFs, method, genetic_initialization,
            sample_custom_query_2,
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface,
            method=method)

        if method != 'genetic':
            ans = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_2,
                    total_CFs=total_CFs, desired_class=desired_class)
        else:
            ans = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_2,
                    total_CFs=total_CFs, desired_class=desired_class,
                    initialization=genetic_initialization)

        assert ans is not None
        if method != 'kdtree':
            assert all(
                ans.cf_examples_list[0].final_cfs_df[exp.data_interface.outcome_name].values == [desired_class] * total_CFs)
        else:
            assert all(
                ans.cf_examples_list[0].final_cfs_df_sparse[exp.data_interface.outcome_name].values ==
                [desired_class] * total_CFs)
        assert all(i == desired_class for i in exp.cfs_preds)

        exp.serialize_explainer(method + '.pkl')
        new_exp = ExplainerBase.deserialize_explainer(method + '.pkl')

        if method != 'genetic':
            ans = new_exp.generate_counterfactuals(
                    query_instances=sample_custom_query_2,
                    total_CFs=total_CFs, desired_class=desired_class)
        else:
            ans = new_exp.generate_counterfactuals(
                    query_instances=sample_custom_query_2,
                    total_CFs=total_CFs, desired_class=desired_class,
                    initialization=genetic_initialization)

        assert ans is not None
        if method != 'kdtree':
            assert all(
                ans.cf_examples_list[0].final_cfs_df[
                    new_exp.data_interface.outcome_name].values == [desired_class] * total_CFs)
        else:
            assert all(
                ans.cf_examples_list[0].final_cfs_df_sparse[new_exp.data_interface.outcome_name].values ==
                [desired_class] * total_CFs)
        assert all(i == desired_class for i in new_exp.cfs_preds)

    @pytest.mark.parametrize(
        ("desired_class", "stopping_threshold", "model_score", "expected_target_score"),
        [(1, 0.3, np.array([0.45, 0.35, 0.2]), 0.35)],
    )
    def test_multiclass_threshold_logic_uses_requested_class_score(
            self, desired_class, stopping_threshold, model_score, expected_target_score, method,
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface,
            method=method)
        exp.num_output_nodes = 3

        exp.misc_init(
            stopping_threshold=stopping_threshold,
            desired_class=desired_class,
            desired_range=None,
            test_pred=np.array([0.2, 0.5, 0.3]),
        )

        assert exp.get_target_class_score(model_score) == pytest.approx(expected_target_score)
        assert exp.is_cf_valid(model_score) is True

    # When no elements in the desired_class are present in the training data
    @pytest.mark.parametrize(("desired_class", "total_CFs"), [(100, 3), ('opposite', 3)])
    def test_unsupported_multiclass(
            self, desired_class, total_CFs, method, sample_custom_query_4,
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface,
            method=method)
        with pytest.raises(UserConfigValidationException) as ucve:
            exp.generate_counterfactuals(query_instances=sample_custom_query_4, total_CFs=total_CFs,
                                         desired_class=desired_class)
        if desired_class == 100:
            assert "Desired class not present in training data!" in str(ucve)
        else:
            assert "Desired class cannot be opposite if the number of classes is more than 2." in str(ucve)

    # Testing for 0 CFs needed
    @pytest.mark.parametrize(("features_to_vary", "desired_class", "desired_range", "total_CFs", "permitted_range"),
                             [("all", 0, None, 0, None)])
    def test_zero_cfs_internal(
            self, method, features_to_vary, desired_class, desired_range, sample_custom_query_2, total_CFs,
            permitted_range, custom_public_data_interface, sklearn_multiclass_classification_model_interface):
        if method == 'genetic':
            pytest.skip('DiceGenetic explainer does not handle the total counterfactuals as zero')
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface,
            method=method)
        features_to_vary = exp.setup(features_to_vary, None, sample_custom_query_2, "inverse_mad")
        exp._generate_counterfactuals(features_to_vary=features_to_vary, query_instance=sample_custom_query_2,
                                      total_CFs=total_CFs, desired_class=desired_class,
                                      desired_range=desired_range, permitted_range=permitted_range)

    @pytest.mark.parametrize("desired_class", [1])
    def test_cfs_type_consistency(
            self, desired_class, method, sample_custom_query_1,
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_multiclass_classification_model_interface,
            method=method)
        cf_explanations = exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=2,
                    desired_class=desired_class)

        for col in sample_custom_query_1.columns:
            assert cf_explanations.cf_examples_list[0].test_instance_df[col].dtype == sample_custom_query_1[col].dtype
            if cf_explanations.cf_examples_list[0].final_cfs_df is not None:
                assert cf_explanations.cf_examples_list[0].final_cfs_df[col].dtype == sample_custom_query_1[col].dtype
            if cf_explanations.cf_examples_list[0].final_cfs_df_sparse is not None:
                assert cf_explanations.cf_examples_list[0].final_cfs_df_sparse[col].dtype == sample_custom_query_1[col].dtype


class TestExplainerBaseRegression:

    @pytest.mark.parametrize(("desired_range", "regression_exp_object"),
                             [([10, 100], 'random'), ([10, 100], 'genetic'), ([10, 100], 'kdtree')],
                             indirect=['regression_exp_object'])
    def test_zero_totalcfs(self, desired_range, regression_exp_object, sample_custom_query_1):
        exp = regression_exp_object  # explainer object
        with pytest.raises(UserConfigValidationException):
            exp.generate_counterfactuals(
                    query_instances=[sample_custom_query_1],
                    total_CFs=0,
                    desired_range=desired_range)

    @pytest.mark.parametrize(("desired_range", "method"),
                             [([3, 5], 'random')])
    def test_numeric_categories(self, desired_range, method):
        x_train, x_test, y_train, y_test, feature_names = \
            create_housing_data()

        x_train = pd.DataFrame(data=x_train, columns=feature_names)
        x_test = pd.DataFrame(data=x_test, columns=feature_names)

        rfc = RandomForestRegressor(n_estimators=10, max_depth=4,
                                    random_state=777)
        model = rfc.fit(x_train, y_train)

        dataset_train = x_train.copy()
        dataset_train['Outcome'] = y_train

        d = dice_ml.Data(dataframe=dataset_train, continuous_features=feature_names, outcome_name='Outcome')
        m = dice_ml.Model(model=model, backend='sklearn', model_type='regressor')
        exp = dice_ml.Dice(d, m, method=method)

        cf_explanation = exp.generate_counterfactuals(
            query_instances=x_test.iloc[0:1],
            total_CFs=10,
            desired_range=desired_range)

        assert cf_explanation is not None

        exp.serialize_explainer("explainer.pkl")
        new_exp = ExplainerBase.deserialize_explainer("explainer.pkl")

        cf_explanation = new_exp.generate_counterfactuals(
            query_instances=x_test.iloc[0:1],
            total_CFs=10,
            desired_range=desired_range)

        assert cf_explanation is not None


class TestExplainerBase:

    def test_instantiating_explainer_base(self, public_data_object):
        with pytest.raises(TypeError):
            ExplainerBase(data_interface=public_data_object)


@pytest.mark.parametrize("method", ['random', 'genetic', 'kdtree'])
class TestExplainerBaseUserConfigValidations:

    @pytest.mark.parametrize('explainer_function',
                             ['generate_counterfactuals', 'local_feature_importance',
                              'feature_importance', 'global_feature_importance'])
    def test_generate_counterfactuals_user_config_validations(
            self, method, sample_custom_query_2,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            explainer_function):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)

        explainer_function = getattr(exp, explainer_function)

        regex_pattern = re.escape(
            'The query instance(s) should not have any missing values. '
            'Please impute the missing values and try again.')

        query_instances_missing_values_numerical = pd.DataFrame({'Categorical': ['a'], 'Numerical': [np.nan]})
        with pytest.raises(
                UserConfigValidationException,
                match=regex_pattern):
            explainer_function(
                query_instances=query_instances_missing_values_numerical, desired_class='opposite',
                total_CFs=10)

        query_instances_missing_values_categorical = pd.DataFrame({'Categorical': [np.nan], 'Numerical': [1]})
        with pytest.raises(
                UserConfigValidationException,
                match=regex_pattern):
            explainer_function(
                query_instances=query_instances_missing_values_categorical, desired_class='opposite',
                total_CFs=10)

        with pytest.raises(
                UserConfigValidationException,
                match=r"The number of counterfactuals generated per query instance \(total_CFs\) "
                      "should be a positive integer."):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=-10, desired_class='opposite')

        with pytest.raises(
                UserConfigValidationException,
                match=r"The number of counterfactuals generated per query instance \(total_CFs\) "
                      "should be a positive integer."):
            explainer_function(
                    query_instances=sample_custom_query_2,
                    total_CFs=0,
                    desired_class="opposite")

        with pytest.raises(
                UserConfigValidationException,
                match=r"The posthoc_sparsity_algorithm should be linear or binary and not random"):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=10,
                               posthoc_sparsity_algorithm='random')

        with pytest.raises(
                UserConfigValidationException,
                match=r"The posthoc_sparsity_algorithm should be linear or binary and not random"):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=10,
                               posthoc_sparsity_algorithm='random')

        with pytest.raises(
                UserConfigValidationException,
                match=r'The stopping_threshold should lie between 0.0 and 1.0'):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=10,
                               stopping_threshold=-10.0)

        with pytest.raises(
                UserConfigValidationException,
                match=r'The posthoc_sparsity_param should lie between 0.0 and 1.0'):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=10,
                               posthoc_sparsity_param=-10.0)

        with pytest.raises(
                UserConfigValidationException,
                match=r'The desired_range parameter should not be set for classification task'):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=10, desired_range=[0, 10])

        with pytest.raises(
                UserConfigValidationException,
                match=r'The desired_class_probability_delta should lie between 0.0 and 1.0'):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=10, desired_class_probability_delta=-0.1)

        with pytest.raises(
                UserConfigValidationException,
                match=r'The desired_class_probability_delta parameter cannot be combined with stopping_threshold.'):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=10, stopping_threshold=0.6,
                               desired_class_probability_delta=0.1)

        with pytest.raises(
                UserConfigValidationException,
                match=r'Some features need to be varied for generating counterfactuals.'):
            explainer_function(query_instances=sample_custom_query_2,
                               total_CFs=10, features_to_vary=[])

    @pytest.mark.parametrize('explainer_function',
                             ['generate_counterfactuals', 'local_feature_importance',
                              'feature_importance', 'global_feature_importance'])
    def test_generate_counterfactuals_user_config_validations_regression(
            self, regression_exp_object, sample_custom_query_1,
            method, explainer_function):
        explainer_function = getattr(regression_exp_object, explainer_function)
        with pytest.raises(
                UserConfigValidationException,
                match=r'The desired_range parameter should be set for regression task'):
            explainer_function(query_instances=sample_custom_query_1,
                               total_CFs=10)

        with pytest.raises(
                UserConfigValidationException,
                match=r'The desired_class_probability_delta parameter should not be set for regression task'):
            explainer_function(query_instances=sample_custom_query_1,
                               total_CFs=10, desired_range=[1, 2.8],
                               desired_class_probability_delta=0.1)

        with pytest.raises(
                UserConfigValidationException,
                match=r'The parameter desired_range needs to have two numbers in ascending order.'):
            explainer_function(query_instances=sample_custom_query_1,
                               total_CFs=10, desired_range=[1, 3, 4])

        with pytest.raises(
                UserConfigValidationException,
                match=r'The range provided in desired_range should be in ascending order.'):
            explainer_function(query_instances=sample_custom_query_1,
                               total_CFs=10, desired_range=[4, 3])


@pytest.mark.parametrize("method", ['random', 'genetic', 'kdtree'])
class TestExplainerBaseDataValidations:
    def test_global_feature_importance_error_conditions_with_insufficient_query_points(
            self, method,
            sample_custom_query_1,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)

        cf_explanations = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_1,
                    total_CFs=15)

        with pytest.raises(
            UserConfigValidationException,
            match="The number of points for which counterfactuals generated should be "
                  "greater than or equal to 10 "
                  "to compute global feature importance"):
            exp.global_feature_importance(
                query_instances=None,
                cf_examples_list=cf_explanations.cf_examples_list)

        with pytest.raises(
            UserConfigValidationException,
            match="The number of query instances should be greater than or equal to 10 "
                  "to compute global feature importance over all query points"):
            exp.global_feature_importance(
                query_instances=sample_custom_query_1,
                total_CFs=15)

    @pytest.mark.skip(reason="Need to fix this test")
    def test_global_feature_importance_error_conditions_with_insufficient_cfs_per_query_point(
            self, method,
            sample_custom_query_10,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)

        cf_explanations = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_10,
                    total_CFs=1)

        with pytest.raises(
            UserConfigValidationException,
            match="The number of counterfactuals generated per query instance should be "
                  "greater than or equal to 10 "
                  "to compute global feature importance over all query points"):
            exp.global_feature_importance(
                query_instances=None,
                cf_examples_list=cf_explanations.cf_examples_list)

        with pytest.raises(
            UserConfigValidationException,
            match="The number of counterfactuals requested per query instance should be greater "
                  "than or equal to 10 "
                  "to compute global feature importance over all query points"):
            exp.global_feature_importance(
                query_instances=sample_custom_query_10,
                total_CFs=1)

    def test_local_feature_importance_error_conditions_with_insufficient_cfs_per_query_point(
            self, method,
            sample_custom_query_1,
            custom_public_data_interface,
            sklearn_binary_classification_model_interface):
        if method == 'genetic':
            pytest.skip('Skipping this test for genetic explainer')

        exp = dice_ml.Dice(
            custom_public_data_interface,
            sklearn_binary_classification_model_interface,
            method=method)

        cf_explanations = exp.generate_counterfactuals(
                    query_instances=sample_custom_query_1,
                    total_CFs=1)

        with pytest.raises(
            UserConfigValidationException,
            match="The number of counterfactuals generated per query instance should be "
                  "greater than or equal to 10 to compute feature importance for all query points"):
            exp.local_feature_importance(
                query_instances=None,
                cf_examples_list=cf_explanations.cf_examples_list)

        with pytest.raises(
            UserConfigValidationException,
            match="The number of counterfactuals requested per "
                  "query instance should be greater than or equal to 10 "
                  "to compute feature importance for all query points"):
            exp.local_feature_importance(
                query_instances=sample_custom_query_1,
                total_CFs=1)
