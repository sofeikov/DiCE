import numpy as np
import pytest
from raiutils.exceptions import UserConfigValidationException

import dice_ml
from dice_ml.counterfactual_explanations import CounterfactualExplanations
from dice_ml.utils import helpers

torch = pytest.importorskip("torch")


@pytest.fixture(scope='session')
def pyt_exp_object():
    backend = 'PYT'
    dataset = helpers.load_adult_income_dataset()
    d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
    ML_modelpath = helpers.get_adult_income_modelpath(backend=backend)
    m = dice_ml.Model(model_path=ML_modelpath, backend=backend, func="ohe-min-max")
    exp = dice_ml.Dice(d, m, method="gradient")
    return exp


class TestDiceTorchMethods:
    @pytest.fixture(autouse=True)
    def _initiate_exp_object(self, pyt_exp_object, sample_adultincome_query):
        self.exp = pyt_exp_object  # explainer object
        self.exp.num_output_nodes = self.exp.model.get_num_output_nodes(
            len(self.exp.data_interface.ohe_encoded_feature_names)
        ).shape[1]
        # initialize required params for CF computations
        self.exp.do_cf_initializations(total_CFs=4, algorithm="DiverseCF", features_to_vary="all")

        # prepare query isntance for CF optimization
        # query_instance = self.exp.data_interface.prepare_query_instance(
        #     query_instance=sample_adultincome_query, encoding='one-hot')
        # self.query_instance = query_instance.iloc[0].values
        self.query_instance = self.exp.data_interface.get_ohe_min_max_normalized_data(
                sample_adultincome_query).iloc[0].to_numpy(dtype=np.float64)

        self.exp.initialize_CFs(self.query_instance, init_near_query_instance=True)  # initialize CFs
        self.exp.target_cf_class = 1  # set desired class to 1

        # setting random feature weights
        np.random.seed(42)
        weights = np.random.rand(len(self.exp.data_interface.ohe_encoded_feature_names))
        self.exp.feature_weights_list = torch.tensor(weights)

    @pytest.mark.parametrize(("yloss", "output"), [("hinge_loss", 10.8252), ("l2_loss", 0.9999), ("log_loss", 9.8253)])
    def test_yloss(self, yloss, output):
        self.exp.yloss_type = yloss
        loss1 = self.exp.compute_yloss()
        assert pytest.approx(loss1.data.detach().numpy(), abs=1e-4) == output

    def test_proximity_loss(self):
        self.exp.x1 = torch.tensor(self.query_instance)
        loss2 = self.exp.compute_proximity_loss()
        # proximity loss computed for given query instance and feature weights.
        assert pytest.approx(loss2.data.detach().numpy(), abs=1e-4) == 0.0068

    @pytest.mark.parametrize(("diversity_loss", "output"), [("dpp_style:inverse_dist", 0.0104), ("avg_dist", 0.1743)])
    def test_diversity_loss(self, diversity_loss, output):
        self.exp.diversity_loss_type = diversity_loss
        loss3 = self.exp.compute_diversity_loss()
        assert pytest.approx(loss3.data.detach().numpy(), abs=1e-4) == output

    def test_regularization_loss(self):
        loss4 = self.exp.compute_regularization_loss()
        # regularization loss computed for given query instance and feature weights.
        assert pytest.approx(loss4.data.detach().numpy(), abs=1e-4) == 0.2086

    def test_final_cfs_and_preds(self, sample_adultincome_query):
        """
        Tets correctness of final CFs and their predictions for sample query instance.
        """
        counterfactual_explanations = self.exp.generate_counterfactuals(
            sample_adultincome_query, total_CFs=4, desired_class="opposite")
        assert isinstance(counterfactual_explanations, CounterfactualExplanations)
        # test_cfs = [[72.0, 'Private', 'HS-grad', 'Married', 'White-Collar', 'White', 'Female', 45.0, 0.691],
        #             [29.0, 'Private', 'Prof-school', 'Married', 'Service', 'White', 'Male', 45.0, 0.954],
        #             [52.0, 'Private', 'Doctorate', 'Married', 'Service', 'White', 'Female', 45.0, 0.971],
        #             [47.0, 'Private', 'Masters', 'Married', 'Service', 'White', 'Female', 73.0, 0.971]]
        # TODO  The model predictions changed after update to posthoc sparsity. Need to investigate.
        # assert dice_exp.final_cfs_df_sparse.values.tolist() == test_cfs

    @pytest.mark.parametrize(
        ("yloss", "expected_loss"),
        [("hinge_loss", 1.8473), ("l2_loss", 0.49), ("log_loss", 1.2040)],
    )
    def test_multiclass_yloss_uses_requested_class_score(self, monkeypatch, yloss, expected_loss):
        self.exp.total_CFs = 1
        self.exp.cfs = [torch.tensor(self.query_instance).float()]
        self.exp.target_cf_class = 1
        self.exp.num_output_nodes = 3
        self.exp.yloss_type = yloss

        monkeypatch.setattr(
            self.exp,
            "get_model_output",
            lambda *args, **kwargs: torch.tensor([0.6, 0.3, 0.1]).float(),
        )

        loss = self.exp.compute_yloss()
        assert pytest.approx(loss.data.detach().numpy(), abs=1e-4) == expected_loss

    @pytest.mark.parametrize(
        ("desired_class", "stopping_threshold", "positive_class_score"),
        [(1, 0.4, 0.43), (0, 0.6, 0.4)],
    )
    def test_respects_user_stopping_threshold_exactly(
        self,
        monkeypatch,
        sample_adultincome_query,
        desired_class,
        stopping_threshold,
        positive_class_score,
    ):
        self.exp.do_cf_initializations(total_CFs=1, algorithm="DiverseCF", features_to_vary="all")
        monkeypatch.setattr(
            self.exp,
            "predict_fn",
            lambda _: np.array([positive_class_score], dtype=np.float32),
        )
        monkeypatch.setattr(self.exp, "stop_loop", lambda *_: True)

        counterfactual_explanations = self.exp.generate_counterfactuals(
            sample_adultincome_query,
            total_CFs=1,
            desired_class=desired_class,
            stopping_threshold=stopping_threshold,
            posthoc_sparsity_param=0,
        )

        final_cfs_df = counterfactual_explanations.cf_examples_list[0].final_cfs_df
        cf_metadata = counterfactual_explanations.cf_examples_list[0].metadata
        assert len(final_cfs_df) == 1
        assert self.exp.stopping_threshold == pytest.approx(stopping_threshold)
        assert final_cfs_df[self.exp.data_interface.outcome_name].iloc[0] == pytest.approx(
            positive_class_score, abs=1e-4
        )
        assert cf_metadata["best_effort_enabled"] is False
        assert cf_metadata["counterfactual_is_valid"] == [True]
        assert cf_metadata["counterfactual_status"] == ["valid"]
        assert cf_metadata["counterfactual_target_scores"] == pytest.approx(
            [self.exp.get_target_class_score(np.array([positive_class_score], dtype=np.float32))],
            abs=1e-4,
        )

    def test_supports_desired_class_probability_delta(
        self,
        monkeypatch,
        sample_adultincome_query,
    ):
        self.exp.do_cf_initializations(total_CFs=1, algorithm="DiverseCF", features_to_vary="all")
        scores = [
            np.array([0.36], dtype=np.float32),
            np.array([0.43], dtype=np.float32),
        ]

        def fake_predict_fn(_):
            if scores:
                return scores.pop(0)
            return np.array([0.43], dtype=np.float32)

        monkeypatch.setattr(self.exp, "predict_fn", fake_predict_fn)
        monkeypatch.setattr(self.exp, "stop_loop", lambda *_: True)

        counterfactual_explanations = self.exp.generate_counterfactuals(
            sample_adultincome_query,
            total_CFs=1,
            desired_class=1,
            desired_class_probability_delta=0.07,
            posthoc_sparsity_param=0,
        )

        cf_metadata = counterfactual_explanations.cf_examples_list[0].metadata
        assert self.exp.stopping_threshold == pytest.approx(0.43)
        assert cf_metadata["desired_class_probability_delta"] == pytest.approx(0.07)
        assert cf_metadata["counterfactual_target_scores"] == pytest.approx([0.43], abs=1e-4)
        assert cf_metadata["counterfactual_is_valid"] == [True]

    def test_multiclass_output_preserves_predicted_class_semantics(
        self,
        monkeypatch,
        sample_adultincome_query,
    ):
        self.exp.do_cf_initializations(total_CFs=1, algorithm="DiverseCF", features_to_vary="all")
        original_num_output_nodes = self.exp.num_output_nodes
        self.exp.num_output_nodes = 3
        try:
            monkeypatch.setattr(
                self.exp,
                "predict_fn",
                lambda _: np.array([0.6, 0.3, 0.1], dtype=np.float32),
            )
            monkeypatch.setattr(self.exp, "stop_loop", lambda *_: True)

            counterfactual_explanations = self.exp.generate_counterfactuals(
                sample_adultincome_query,
                total_CFs=1,
                desired_class=0,
                stopping_threshold=0.5,
                posthoc_sparsity_param=0,
            )

            final_cfs_df = counterfactual_explanations.cf_examples_list[0].final_cfs_df
            assert len(final_cfs_df) == 1
            assert final_cfs_df[self.exp.data_interface.outcome_name].iloc[0] == 0
        finally:
            self.exp.num_output_nodes = original_num_output_nodes

    def test_unreachable_threshold_requires_explicit_best_effort_label(
        self,
        monkeypatch,
        sample_adultincome_query,
    ):
        self.exp.do_cf_initializations(total_CFs=1, algorithm="DiverseCF", features_to_vary="all")
        monkeypatch.setattr(
            self.exp,
            "predict_fn",
            lambda _: np.array([0.43], dtype=np.float32),
        )
        monkeypatch.setattr(self.exp, "stop_loop", lambda *_: True)

        with pytest.raises(UserConfigValidationException, match="No counterfactuals found for any of the query points"):
            self.exp.generate_counterfactuals(
                sample_adultincome_query,
                total_CFs=1,
                desired_class=1,
                stopping_threshold=0.8,
                posthoc_sparsity_param=0,
            )

    @pytest.mark.parametrize(
        ("permitted_direction", "expected_comparison"),
        [
            ({"hours_per_week": "increase"}, "ge"),
            ({"hours_per_week": "decrease"}, "le"),
        ],
    )
    def test_respects_permitted_direction(
        self,
        sample_adultincome_query,
        permitted_direction,
        expected_comparison,
    ):
        counterfactual_explanations = self.exp.generate_counterfactuals(
            sample_adultincome_query,
            total_CFs=1,
            desired_class="opposite",
            features_to_vary=["hours_per_week"],
            permitted_direction=permitted_direction,
            min_iter=0,
            max_iter=1,
            posthoc_sparsity_param=0,
            best_effort=True,
        )

        final_cfs_df = counterfactual_explanations.cf_examples_list[0].final_cfs_df
        assert final_cfs_df is not None

        query_value = sample_adultincome_query["hours_per_week"].iloc[0]
        counterfactual_value = final_cfs_df["hours_per_week"].iloc[0]
        if expected_comparison == "ge":
            assert counterfactual_value >= query_value
        else:
            assert counterfactual_value <= query_value

    @pytest.mark.parametrize("version", ["1.0", "2.0"])
    def test_best_effort_metadata_survives_serialization(
        self,
        monkeypatch,
        sample_adultincome_query,
        version,
    ):
        self.exp.do_cf_initializations(total_CFs=1, algorithm="DiverseCF", features_to_vary="all")
        monkeypatch.setattr(
            self.exp,
            "predict_fn",
            lambda _: np.array([0.43], dtype=np.float32),
        )
        monkeypatch.setattr(self.exp, "stop_loop", lambda *_: True)

        counterfactual_explanations = self.exp.generate_counterfactuals(
            sample_adultincome_query,
            total_CFs=1,
            desired_class=1,
            stopping_threshold=0.8,
            posthoc_sparsity_param=0,
            best_effort=True,
        )
        counterfactual_explanations.metadata["version"] = version

        recovered_counterfactual_explanations = CounterfactualExplanations.from_json(
            counterfactual_explanations.to_json()
        )
        recovered_metadata = recovered_counterfactual_explanations.cf_examples_list[0].metadata
        final_cfs_df = recovered_counterfactual_explanations.cf_examples_list[0].final_cfs_df

        assert final_cfs_df[self.exp.data_interface.outcome_name].iloc[0] == pytest.approx(0.43, abs=1e-4)

        assert recovered_metadata["best_effort_enabled"] is True
        assert recovered_metadata["counterfactual_is_valid"] == [False]
        assert recovered_metadata["counterfactual_status"] == ["best_effort"]
        assert recovered_metadata["counterfactual_target_scores"] == pytest.approx([0.43], abs=1e-4)
        assert recovered_metadata["stopping_threshold"] == pytest.approx(0.8)
        assert recovered_metadata["target_class"] == 1
