"""Run dependency-free assertions when pytest is unavailable."""
from student_code.tests import test_ilr_objective as ilr
from student_code.tests import test_ilr_data as ilr_data
from student_code.tests import test_qmt_model as qmt


def main():
    checks = [
        ilr.test_first_answer_score_uses_pre_answer_logits_and_keeps_gradient,
        ilr.test_ilr_prefers_the_higher_iou_candidate,
        ilr.test_combined_loss_reports_all_components,
        ilr.test_zero_auxiliary_weights_is_pair_grouped_ce_ablation,
        ilr_data.test_fixed_pair_pool_matches_ilr_contract,
        qmt.test_qmt_configuration_is_checkpoint_reloadable,
        qmt.test_qmt_parameters_receive_updates,
        qmt.test_hard_negative_margin_uses_gt_interior_and_exterior,
        qmt.test_qmt_dev_evaluation_uses_row_contract_and_brief_map,
    ]
    for check in checks:
        check()
    print(f"{len(checks)} quick checks passed")


if __name__ == "__main__":
    main()
