from python_model import PythonFunction

MODULE = "bentoml.picklable_model"
MODEL_NAME = "py_model.case-1.grpc.e2e"

model = PythonFunction()

signatures = {
    "predict_file": {"batchable": True},
    "echo_json": {"batchable": True},
    "echo_object": {"batchable": False},
    "echo_ndarray": {"batchable": True},
    "double_ndarray": {"batchable": True},
    "multiply_float_ndarray": {"batchable": True},
    "double_dataframe_column": {"batchable": True},
}
