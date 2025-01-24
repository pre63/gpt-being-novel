import networkx as nx


class DynamicTaskPrioritizer:
  def __init__(self, tasks, dependencies):
    self.graph = nx.DiGraph()
    for task in tasks:
      self.graph.add_node(task, priority=0.5)
    self.graph.add_edges_from(dependencies)

  def update_priorities(self, feedback):
    for task in self.graph.nodes:
      self.graph.nodes[task]["priority"] += feedback.get(task, 0)
      self.graph.nodes[task]["priority"] = max(0, min(1, self.graph.nodes[task]["priority"]))

  def prioritize(self):
    sorted_tasks = sorted(self.graph.nodes, key=lambda t: self.graph.nodes[t]["priority"], reverse=True)
    resolved_tasks = []
    for task in sorted_tasks:
      if all(dep in resolved_tasks for dep in self.graph.predecessors(task)):
        resolved_tasks.append(task)
    return resolved_tasks
