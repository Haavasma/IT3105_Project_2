import tensorflow as tf
import numpy as np


class LiteModel:
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()
        self.input_index = input_det["index"]
        self.policy_index = output_det[0]["index"]
        self.value_index = output_det[1]["index"]
        self.input_shape = input_det["shape"]
        self.policy_shape = output_det[0]["shape"]
        self.value_shape = output_det[1]["shape"]
        self.input_dtype = input_det["dtype"]
        self.policy_dtype = output_det[0]["dtype"]
        self.value_dtype = output_det[1]["dtype"]

    def predict_policy(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.policy_shape[1]), dtype=self.policy_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i : i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.policy_index)[0]
        return out

    def predict_value(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.value_shape[1]), dtype=self.value_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i : i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.value_index)[0]
        return out

    def predict_single_policy(self, inp):
        """Like predict(), but only for a single record. The input data can be a Python list."""
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.policy_index)

        return out[0]

    def predict_single_value(self, inp):
        """Like predict(), but only for a single record. The input data can be a Python list."""
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.value_index)

        return out[0]
