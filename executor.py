from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import torch
import tensorflow as tf
from values import Value
from operations import OperationInstance
from ml_types import MatrixInstance


def initialize_seed_arrays(
    seed_values: list[Value],
    rng_seed: int = 84,
    low: float = -2.0,
    high: float = 2.0,
    np_dtype: np.dtype = np.float64,
    p_nan: float = 0.05,   # probability of injecting a NaN
    p_inf: float = 0.02,   # probability of injecting an Inf
) -> dict[str, np.ndarray]:



    rng = np.random.default_rng(rng_seed)
    initial_arrays: dict[str, np.ndarray] = {}

    for seed in seed_values:
        if seed.shape is None:
            raise ValueError(f"Seed value {seed.name} has no shape.")
        if seed.name in initial_arrays:
            raise ValueError(f"Duplicate seed name detected: {seed.name}")

        rows, cols = seed.shape

        # --- base generation ---
        if seed.matrix_type == MatrixInstance.Symmetric:
            if rows != cols:
                raise ValueError(
                    f"Symmetric matrix must be square: {seed.name} has shape {seed.shape}"
                )
            base = rng.uniform(low=low, high=high, size=(rows, cols)).astype(np_dtype)
            arr = (base + base.T) / 2
        else:
            arr = rng.uniform(low=low, high=high, size=(rows, cols)).astype(np_dtype)

        # --- inject NaN / Inf (controlled) ---
        if rng.random() < p_nan:
            i = rng.integers(0, rows)
            j = rng.integers(0, cols)

            if seed.matrix_type == MatrixInstance.Symmetric and rows == cols:
                arr[i, j] = np.nan
                arr[j, i] = np.nan  # preserve symmetry
            else:
                arr[i, j] = np.nan

        if rng.random() < p_inf:
            i = rng.integers(0, rows)
            j = rng.integers(0, cols)
            val = np.inf if rng.random() < 0.5 else -np.inf

            if seed.matrix_type == MatrixInstance.Symmetric and rows == cols:
                arr[i, j] = val
                arr[j, i] = val
            else:
                arr[i, j] = val

        initial_arrays[seed.name] = arr.astype(np_dtype)

    return initial_arrays
class SequenceExecutor:
    def __init__(
        self,
        seed_values: list[Value],
        ops_applied: list[OperationInstance],
        framework: str,
        initial_arrays: dict[str, np.ndarray],
    ) -> None:
        self.seed_values = seed_values
        self.ops_applied = ops_applied
        self.framework = framework
        self.initial_arrays = self._copy_initial_arrays(initial_arrays)
    # ---------- Initialization ----------

   

    def _copy_initial_arrays(
        self,
        initial_arrays: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        copied: Dict[str, np.ndarray] = {}
        for name, arr in initial_arrays.items():
            copied[name] = np.array(arr, copy=True)
        return copied

    def get_initial_arrays_copy(self) -> Dict[str, np.ndarray]:
        return self._copy_initial_arrays(self.initial_arrays)

    # ---------- Pretty-print helpers ----------

    def _format_symbolic_value(self, v: Value) -> str:
        if v.shape is None:
            return f"{v.name}:{v.type.value}"
        return f"{v.name}:{v.type.value}{v.shape}"

    def _tensor_shape_str(self, tensor: Any) -> str:
        if hasattr(tensor, "shape"):
            return str(tuple(tensor.shape))
        return "scalar"

    def _framework_call_str(self, op_name: str, arg_names: List[str]) -> str:
        if self.framework == "torch":
            if op_name == "Add":
                return f"torch.add({arg_names[0]}, {arg_names[1]})"
            if op_name == "Subtract":
                return f"torch.sub({arg_names[0]}, {arg_names[1]})"
            if op_name == "MatMul":
                return f"torch.matmul({arg_names[0]}, {arg_names[1]})"
            if op_name == "Transpose":
                return f"torch.transpose({arg_names[0]}, 0, 1)"
            if op_name == "Sum":
                return f"torch.sum({arg_names[0]})"
            if op_name == "LogDet":
                return f"torch.linalg.logdet({arg_names[0]})"
            if op_name == "Cholesky":
                return f"torch.linalg.cholesky({arg_names[0]})"
            if op_name == "Solve":
                return f"torch.linalg.solve({arg_names[0]}, {arg_names[1]})"
            if op_name == "Softmax":
                return f"torch.nn.functional.softmax({arg_names[0]}, dim=-1)"

            raise NotImplementedError(f"Unsupported op: {op_name}")


        if self.framework == "tf":
            if op_name == "Add":
                return f"tf.add({arg_names[0]}, {arg_names[1]})"
            if op_name == "Subtract":
                return f"tf.subtract({arg_names[0]}, {arg_names[1]})"
            if op_name == "MatMul":
                return f"tf.linalg.matmul({arg_names[0]}, {arg_names[1]})"
            if op_name == "Transpose":
                return f"tf.transpose({arg_names[0]}, perm=[1, 0])"
            if op_name == "Sum":
                return f"tf.reduce_sum({arg_names[0]})"
            if op_name == "LogDet":
                return f"tf.linalg.logdet({arg_names[0]})"
            if op_name == "Cholesky":
                return f"tf.linalg.cholesky({arg_names[0]})"
            if op_name == "Solve":
                return f"tf.linalg.solve({arg_names[0]}, {arg_names[1]})"
            if op_name == "Softmax":
                return f"tf.nn.softmax({arg_names[0]}, axis=-1)"

            raise NotImplementedError(f"Unsupported op: {op_name}")
           

        raise NotImplementedError(f"Unsupported framework/op: {self.framework}/{op_name}")

        raise NotImplementedError(f"Unsupported framework/op: {self.framework}/{op_name}")

    # ---------- Framework env creation ----------

    def _make_env(self) -> Dict[str, Any]:
        if self.framework == "torch":
            return self._make_torch_env()
        if self.framework == "tf":
            return self._make_tf_env()
        raise NotImplementedError(f"Unsupported framework: {self.framework}")

        

    def _make_torch_env(self) -> Dict[str, torch.Tensor]:
        env: Dict[str, torch.Tensor] = {}
        for name, arr in self.initial_arrays.items():
            env[name] = torch.tensor(arr, dtype=torch.float64)
        return env

    def _make_tf_env(self) -> Dict[str, tf.Tensor]:
        env: Dict[str, tf.Tensor] = {}
        for name, arr in self.initial_arrays.items():
            env[name] = tf.convert_to_tensor(arr, dtype=tf.float64)
        return env

    # ---------- Op dispatch ----------

    def _apply_op(self, op_inst: OperationInstance, env: Dict[str, Any]) -> Any:
        op_name = op_inst.operation.name
        args = [env[arg.name] for arg in op_inst.args]

        if self.framework == "torch":
            return self._apply_torch_op(op_name, args)
        if self.framework == "tf":
            return self._apply_tf_op(op_name, args)

        raise NotImplementedError(f"Unsupported framework: {self.framework}")

    def _apply_torch_op(
        self,
        op_name: str,
        args: List[torch.Tensor],
    ) -> torch.Tensor:
        if op_name == "Add":
            return torch.add(args[0], args[1])
        if op_name == "Subtract":
            return torch.sub(args[0], args[1])
        if op_name == "MatMul":
            return torch.matmul(args[0], args[1])
        if op_name == "Transpose":
            return torch.transpose(args[0], 0, 1)
        if op_name == "Sum":
            return torch.sum(args[0])
        if op_name == "LogDet":
                return torch.logdet(args[0])
        if op_name == "Cholesky":
            return torch.linalg.cholesky(args[0])

        if op_name == "Solve":
            return torch.linalg.solve(args[0], args[1])

        if op_name == "Softmax":
            return torch.nn.functional.softmax(args[0], dim=-1)
        
        raise NotImplementedError(f"Unsupported op: {op_name}")

    def _apply_tf_op(
        self,
        op_name: str,
        args: List[tf.Tensor],
) -> tf.Tensor:
        if op_name == "Add":
            return tf.add(args[0], args[1])
        if op_name == "Subtract":
            return tf.subtract(args[0], args[1])

        if op_name == "MatMul":
            return tf.linalg.matmul(args[0], args[1])

        if op_name == "Transpose":
            return tf.transpose(args[0], perm=[1, 0])

        if op_name == "Sum":
            return tf.reduce_sum(args[0])

        if op_name == "LogDet":
            return tf.linalg.logdet(args[0])

        if op_name == "Cholesky":
            return tf.linalg.cholesky(args[0])

        if op_name == "Solve":
            return tf.linalg.solve(args[0], args[1])

        if op_name == "Softmax":
            return tf.nn.softmax(args[0], axis=-1)
        
        raise NotImplementedError(f"Unsupported op: {op_name}")

        

    # ---------- Execution ----------

    def execute(self, verbose: bool = False) -> Dict[str, Any]:
        env = self._make_env()

        if verbose:
            print(f"\nFramework: {self.framework}")
            print("Beginning symbolic replay...")

        for step_idx, op_inst in enumerate(self.ops_applied):
            arg_names = [arg.name for arg in op_inst.args]
            symbolic_args = ", ".join(self._format_symbolic_value(arg) for arg in op_inst.args)
            temp_name = f"t{step_idx}"

            result = self._apply_op(op_inst, env)
            env[temp_name] = result

            if verbose:
                print(f"\nStep {step_idx}")
                print(f"  Symbolic: {op_inst.operation.name}({symbolic_args}) -> {temp_name}")
                print(f"  Concrete: {self._framework_call_str(op_inst.operation.name, arg_names)}")
                print(f"  Result shape: {self._tensor_shape_str(result)}")
                print(f"  Result value:\n{result}")

        return env

    def execute_final(self, verbose: bool = False) -> Any:
        if not self.ops_applied:
            raise ValueError("No operations to execute.")

        env = self.execute(verbose=verbose)
        final_name = f"t{len(self.ops_applied) - 1}"
        return env[final_name]

    # ---------- Debug helpers ----------

    def format_initial_values(self) -> str:
        lines = []
        lines.append(f"Framework: {self.framework}")
        lines.append("Initial concrete seed values:")
        for name, arr in self.initial_arrays.items():
            lines.append(f"{name}: shape={arr.shape}")
            lines.append(str(arr))
            lines.append("")
        return "\n".join(lines)

    def format_execution_trace(self) -> str:
        env = self._make_env()
        lines = []

        lines.append(f"Framework: {self.framework}")
        lines.append("Beginning symbolic replay...")

        for step_idx, op_inst in enumerate(self.ops_applied):
            arg_names = [arg.name for arg in op_inst.args]
            symbolic_args = ", ".join(self._format_symbolic_value(arg) for arg in op_inst.args)
            temp_name = f"t{step_idx}"

            result = self._apply_op(op_inst, env)
            env[temp_name] = result

            lines.append("")
            lines.append(f"Step {step_idx}")
            lines.append(f"  Symbolic: {op_inst.operation.name}({symbolic_args}) -> {temp_name}")
            lines.append(f"  Concrete: {self._framework_call_str(op_inst.operation.name, arg_names)}")
            lines.append(f"  Result shape: {self._tensor_shape_str(result)}")
            lines.append(f"  Result value:\n{result}")

        final_name = f"t{len(self.ops_applied) - 1}"
        lines.append("")
        lines.append(f"Framework: {self.framework}")
        lines.append("Final result:")
        lines.append(str(env[final_name]))

        return "\n".join(lines)

    def format_final_env(self) -> str:
        env = self.execute()

        lines = []
        lines.append(f"Framework: {self.framework}")
        lines.append("Final Environment (all values):")

        for name, val in env.items():
            # shape handling
            if hasattr(val, "shape"):
                shape_str = str(tuple(val.shape))
            else:
                shape_str = "scalar"

            lines.append(f"{name}: shape={shape_str}")
            lines.append(str(val))
            lines.append("")

        return "\n".join(lines)

    def print_initial_arrays(self) -> None:
        print(self.format_initial_arrays())

    def print_final_result(self) -> None:
        print(self.format_execution_trace())

    def print_final_env(self):
        print(self.format_final_env())