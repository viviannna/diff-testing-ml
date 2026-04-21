import random
from values import Value
from operations import OperationInstance


class Printer:
    rng: random.Random

    def __init__(self, rng: random.Random):
        self.rng = rng

    def format_value(self, v: Value) -> str:
        if v.shape is None:
            return f"{v.name}:{v.type.value}"
        return f"{v.name}:{v.type.value}{v.shape}"


    # def print_generated_seq(self, ops_applied: List[OperationInstance], values: List[Value], og_values: List[Value]):
    #     print(f"Random Seed: {self.rng.seed}:")
    #     print("Original Seed Values:")
    #     for v in og_values:
    #         print(f" {self.format_value(v)}")

    #     print("Generated Sequence:")
    #     output_values = values[len(og_values):]
    #     for i, (op, out_val) in enumerate(zip(ops_applied, output_values)):
    #         args_str = ", ".join(self.format_value(a) for a in op.args)
    #         print(f" {i}: {op.operation.name}({args_str}) -> {self.format_value(out_val)}")

    #     print("Existing Values:")
    #     for v in values:
    #         print(f" {self.format_value(v)}")

    def format_generated_seq(self, ops_applied, values, og_values) -> str:
        lines = []
        lines.append(f"Random Seed: {self.rng.seed}:")
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

