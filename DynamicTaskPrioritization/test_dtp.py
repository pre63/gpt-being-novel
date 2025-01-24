from dtp_model import DynamicTaskPrioritizer

# Define tasks and dependencies
tasks = ["TaskA", "TaskB", "TaskC", "TaskD"]
dependencies = [("TaskA", "TaskB"), ("TaskB", "TaskC"), ("TaskC", "TaskD")]

# Initialize the prioritizer
dtp = DynamicTaskPrioritizer(tasks, dependencies)

# Update priorities with feedback
dtp.update_priorities({"TaskA": 0.2, "TaskB": -0.1, "TaskC": 0.3})

# Debug priorities and dependencies
print("Task priorities after feedback:", {task: dtp.graph.nodes[task]["priority"] for task in dtp.graph.nodes})

# Get the prioritized order
prioritized_tasks = dtp.prioritize()
print("Prioritized Tasks:", prioritized_tasks)

# Debug check
expected_order = ["TaskA", "TaskC", "TaskB", "TaskD"]
print("Test Passed:", prioritized_tasks == expected_order)
