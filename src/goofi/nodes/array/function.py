import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import StringParam


class Function(Node):
    def config_input_slots():
        return {"array": DataType.ARRAY}

    def config_output_slots():
        return {"out": DataType.ARRAY}

    def config_params():
        return {
            "function": {
                "function": StringParam(
                    "abs",
                    options=["abs", "sqrt", "log", "exp", "sin", "cos", "tan", "arcsin", "arccos", "arctan"],
                    doc="Element-wise function to apply to the array data.",
                )
            }
        }

    def process(self, array: Data):
        if array is None:
            return None

        func = getattr(np, self.params.function.function.value, None)
        if func is None:
            raise ValueError(f"Unknown function: {self.params.function.function.value}")
        result = func(array.data)

        return {"out": (result, array.meta)}
