
class Helper:
    rng: random.Random

    def __init__(self, rng: random.Random):
        self.rng = rng

    # Prints out the original seed values and the entire generated sequence
    def print_generated_seq(self, ops_applied: List[OperationInstance], values: List[Value], og_values: List[Value]):
        print(f"Random Seed: {self.rng.seed}:")
        print(f"Original Seed Values:")
        for v in og_values:
            print(f" {v.name}:{v.type.value}")
        print("Generated Sequence:")
        for i, op in enumerate(ops_applied):
            print(f" {i}: {op.operation.name}({', '.join(f'{a.name}:{a.type.value}' for a in op.args)}) -> {op.operation.output_type.value}")

        # want to print out all the existing values
        print("Existing Values:")
        for v in values:
            print(f" {v.name}:{v.type.value}")

    # Prints out current values and pool of legal operations at each step
    def print_step_decisions(self, current_len: int, values: List[Value], legal_ops: List[OperationInstance], op_inst: OperationInstance):
        print(f"\n\nStep {current_len}:")
        print("Available Values:")
        for v in values:
            print(f" {v.name}:{v.type.value}")

        print("Legal Ops: ")
        for i, op in enumerate(legal_ops):
            print(f" {i}: {op.operation.name}({', '.join(f'{a.name}:{a.type.value}' for a in op.args)}) -> {op.operation.output_type.value}")
        
        print("Chosen Op:")
        print(f" {op_inst.operation.name}({', '.join(f'{a.name}:{a.type.value}' for a in op_inst.args)}) -> {op_inst.operation.output_type.value}")

