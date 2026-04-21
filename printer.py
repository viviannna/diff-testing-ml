import random
from values import Value
from operations import OperationInstance


class Printer:
    rng: random.Random

    def __init__(self, rng: random.Random, seed_value: int):
        self.rng = rng
        self.seed_value = seed_value

    def format_value(self, v: Value) -> str:
        matrix_type_str = ""
        if getattr(v, "matrix_type", None) is not None:
            matrix_type_str = f"[{v.matrix_type.value}]"

        if v.shape is None:
            return f"{v.name}:{v.type.value}{matrix_type_str}"

        return f"{v.name}:{v.type.value}{matrix_type_str}{v.shape}"

    def format_generated_seq(self, ops_applied, values, og_values) -> str:
        lines = []
        lines.append(f"Random Seed: {self.seed_value}")
        lines.append("Original Seed Values:")
        for v in og_values:
            lines.append(f" {self.format_value(v)}")

        lines.append("")
        lines.append("Generated Sequence:")
        output_values = values[len(og_values):]
        for i, (op, out_val) in enumerate(zip(ops_applied, output_values)):
            args_str = ", ".join(self.format_value(a) for a in op.args)
            lines.append(f" {i}: {op.operation.name}({args_str}) -> {self.format_value(out_val)}")

        lines.append("")
        lines.append("Existing Values:")
        for v in values:
            lines.append(f" {self.format_value(v)}")

        return "\n".join(lines)

    def print_generated_seq(self, ops_applied, values, og_values):
        print(self.format_generated_seq(ops_applied, values, og_values))



    # Prints out current values and pool of legal operations at each step
    def print_step_decisions(self, current_len: int, values: list[Value], legal_ops: list[OperationInstance], op_inst: OperationInstance):
        print(f"\n\nStep {current_len}:")
        print("Available Values:")
        for v in values:
            print(f" {v.name}:{v.type.value}")

        print("Legal Ops: ")
        for i, op in enumerate(legal_ops):
            print(f" {i}: {op.operation.name}({', '.join(f'{a.name}:{a.type.value}' for a in op.args)}) -> {op.operation.output_type.value}")
        
        print("Chosen Op:")
        print(f" {op_inst.operation.name}({', '.join(f'{a.name}:{a.type.value}' for a in op_inst.args)}) -> {op_inst.operation.output_type.value}")

