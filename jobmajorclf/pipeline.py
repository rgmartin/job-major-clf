import os
import pathlib
from pathlib import Path

import boto3
import sagemaker
from sagemaker import Model, image_uris
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import (FrameworkProcessor, ProcessingInput,
                                  ProcessingOutput, ScriptProcessor)
from sagemaker.sklearn import SKLearn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (ParameterFloat, ParameterInteger,
                                           ParameterString)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_experiment_config import \
    PipelineExperimentConfig
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

PIPELINE_DIR = pathlib.Path(__file__).resolve().parent


# todo: remove below when all fixed
def print_directory_structure(directory_path):
    p = Path(directory_path)
    for f in p.glob("**/*"):
        print(f)


print_directory_structure(PIPELINE_DIR)


def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    sagemaker_project_id=None,
    role=None,
    default_bucket=None,
    input_data_url=None,
    # todo remove default values for bucket and input data
    # default_bucket="rubert-data",
    # input_data_url="s3://rubert-data/job-major-clf/input/",
    bucket_prefix="job-major-clf",
    model_package_group_name="job-major-clf-model-group",
    pipeline_name="job-major-clf-pipeline",
    processing_instance_type="ml.c5.xlarge",
    training_instance_type="ml.m5.xlarge",
    test_score_threshold=0.75,
):
    """Gets a SageMaker ML Pipeline instance.

    Args:
        aws_region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    # Get sessions
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    session = get_pipeline_session(region, default_bucket)

    #    sm = session.sagemaker_client
    # pipeline_session = PipelineSession()
    # role = os.environ.get(
    #    "SM_PIPELINE_ROLE_ARN",
    #    sagemaker.session.get_execution_role(sagemaker_session=pipeline_session),
    # )
    # sm_project_name = os.environ.get("SM_PROJECT_NAME", "local_project")
    # mg_name = os.environ.get("MPG_NAME", "local-model-group")
    # commit = os.environ.get("COMMIT_ID", "0123456789")
    # aws_region = os.environ.get("AWS_DEFAULT_REGION", pipeline_session.boto_region_name)

    # Set S3 urls for processed data
    train_s3_url = f"s3://{default_bucket}/{bucket_prefix}/train"
    test_s3_url = f"s3://{default_bucket}/{bucket_prefix}/test"
    evaluation_s3_url = f"s3://{default_bucket}/{bucket_prefix}/evaluation"

    # Parameters for pipeline execution
    # Set processing instance type
    process_instance_type_param = ParameterString(
        name="ProcessingInstanceType",
        default_value=processing_instance_type,
    )

    # Set training instance type
    train_instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=training_instance_type,
    )

    # Set training instance count
    train_instance_count_param = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )

    # Set model approval param
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    #  Minimal threshold for model performance on the test dataset
    test_score_threshold_param = ParameterFloat(
        name="TestScoreThreshold", default_value=test_score_threshold
    )

    # Set S3 urls for input datasets
    input_s3_param = ParameterString(
        name="InputDataUrl",
        default_value=input_data_url,
    )

    # Define step cache config
    cache_config = CacheConfig(enable_caching=True, expire_after="P30d")  # 30-day

    # ---------------------------------------------------------------------------------------------
    # 1. Processing step for feature engineering
    processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.2-1",
        role=role,
        instance_type=process_instance_type_param,
        instance_count=1,
        base_job_name=f"{pipeline_name}/preprocess",
        sagemaker_session=session,
    )

    processing_inputs = [
        ProcessingInput(source=input_s3_param, destination="/opt/ml/processing/input"),
    ]

    processing_outputs = [
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=train_s3_url,
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=test_s3_url,
        ),
    ]

    step_process = ProcessingStep(
        name=f"{pipeline_name}-preprocess-data",
        cache_config=cache_config,
        step_args=processor.run(
            code="preprocessing.py",
            source_dir=str(PIPELINE_DIR),
            inputs=processing_inputs,
            outputs=processing_outputs,
            arguments=[
                "--filepath",
                "/opt/ml/processing/input/",
                "--filename-alt",
                "Alternate Titles.xlsx",
                "--filename-occ",
                "Occupation Data.xlsx",
                "--filename-soc-structure",
                "soc_structure_2018.csv",
                "--outputpath",
                "/opt/ml/processing/output/",
            ],
        ),
    )

    # --------------------------------------------------------------------------------------
    # Training step for generating model artifacts
    metric_definitions = [
        {"Name": "train_loss", "Regex": r"'loss': ([0-9]+(.|e\-)[0-9]+),?"},
        {"Name": "eval_loss", "Regex": r"'eval_loss': ([0-9]+(.|e\-)[0-9]+),?"},
        {"Name": "eval_accuracy", "Regex": r"'eval_accuracy': ([0-9]+(.|e\-)[0-9]+),?"},
        {"Name": "eval_f1", "Regex": r"'eval_f1': ([0-9]+(.|e\-)[0-9]+),?"},
        {"Name": "eval_runtime", "Regex": r"'eval_runtime': ([0-9]+(.|e\-)[0-9]+),?"},
        {
            "Name": "eval_samples_per_second",
            "Regex": r"'eval_samples_per_second': ([0-9]+(.|e\-)[0-9]+),?",
        },
        {"Name": "epoch", "Regex": r"'epoch': ([0-9]+(.|e\-)[0-9]+),?"},
    ]

    hyperparameters = {
        "epochs": "5",
        "train-batch-size": "32",
        "eval-batch-size": "32",
        "base-model-name": "distilbert-base-uncased",
        "warmup-steps": "50",
        "learning-rate": "2e-5",
        "early-stopping-patience": "3",
    }
    training_image_uri = image_uris.retrieve(
        framework="huggingface",
        region=region,
        version="4.28.1",
        image_scope="training",
        base_framework_version="pytorch2.0.0",
    )

    estimator = HuggingFace(
        entry_point=str("train.py"),
        disable_profiler=True,
        source_dir=str(PIPELINE_DIR),
        base_job_name=f"{pipeline_name}/train",
        output_path=f"s3://{session.default_bucket()}/{pipeline_name}/train",
        # output_kms_key=kms_key_alias,
        sagemaker_session=session,
        role=role,
        metric_definitions=metric_definitions,
        instance_type=train_instance_type_param,
        instance_count=train_instance_count_param,
        py_version="py310",
        image_uri=training_image_uri,
        hyperparameters=hyperparameters,
    )

    train_data = step_process.properties.ProcessingOutputConfig.Outputs[
        "train"
    ].S3Output.S3Uri
    test_data = step_process.properties.ProcessingOutputConfig.Outputs[
        "test"
    ].S3Output.S3Uri

    step_train = TrainingStep(
        name=f"{pipeline_name}-train",
        cache_config=cache_config,
        step_args=estimator.fit(
            inputs={
                "train": TrainingInput(s3_data=train_data),
                "test": TrainingInput(
                    s3_data=test_data,
                ),
            }
        ),
    )
    # --------------------------------------------------------------------------------
    # Evaluation step
    script_processor = ScriptProcessor(
        image_uri=training_image_uri,
        role=role,
        command=["python3"],
        instance_type=process_instance_type_param,
        instance_count=1,
        base_job_name=f"{pipeline_name}/evaluate",
        sagemaker_session=session,
    )

    eval_inputs = [
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source="s3://rubert-data/job-major-clf/test",
            destination="/opt/ml/processing/test",
        ),
    ]

    eval_outputs = [
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=evaluation_s3_url,
        ),
    ]

    eval_args = script_processor.run(
        inputs=eval_inputs,
        outputs=eval_outputs,
        code=f"{str(PIPELINE_DIR)}/evaluation.py",
    )

    evaluation_report = PropertyFile(
        name="ModelEvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    step_eval = ProcessingStep(
        name=f"{pipeline_name}-evaluate-model",
        step_args=eval_args,
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    # ------------------------------------------------------------------
    # Define register step
    model = Model(
        image_uri=training_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=session,
        role=role,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            content_type="application/json",
        )
    )

    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge", "ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status_param,
        model_metrics=model_metrics,
    )

    step_register = ModelStep(name=f"{pipeline_name}-register", step_args=register_args)

    # -----------------------------------------------------------
    # Fail step
    step_fail = FailStep(
        name=f"{pipeline_name}-fail",
        error_message=Join(
            on=" ",
            values=["Execution failed due to f1 Score <", test_score_threshold_param],
        ),
    )
    # ----------------------------------------------------------------------
    # # Condition step
    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.test_f1",
        ),
        right=test_score_threshold_param,
    )

    step_cond = ConditionStep(
        name=f"{pipeline_name}-check-test-score",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            process_instance_type_param,
            train_instance_type_param,
            train_instance_count_param,
            model_approval_status_param,
            test_score_threshold_param,
            input_s3_param,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=session,
    )

    return pipeline
