from PyQt6.QtGui import QColor

class Node:
    def __init__(self, name, shape='circle', color='#ffffff', thickness=1, filled=False, x=0, y=0, text=None):
        self.name = name
        self.shape = shape                # 'circle', 'square', 'text'
        self.text = text                  # if shape = 'text'
        self.color = QColor(color) if isinstance(color, str) else color  
        self.thickness = thickness
        self.filled = filled
        self.x = x
        self.y = y

class SkeletonModel:
    def __init__(self):
        self.nodes = {}
        self.edges = set() 
        self.syms = set() 
        self.node_counter = 0

    def add_node(self, shape='circle', color='#666666', thickness=1, filled=False, x=0, y=0, text=None):
        self.node_counter += 1
        name = f"Node{self.node_counter}"
        while name in self.nodes:
            self.node_counter += 1
            name = f"Node{self.node_counter}"
        node = Node(name, shape, color, thickness, filled, x, y, text)
        self.nodes[name] = node
        return node

    def remove_node(self, name):
        if name not in self.nodes:
            return
        edges_to_remove = [e for e in self.edges if name in e]
        for e in edges_to_remove:
            self.edges.remove(e)
        syms_to_remove = [e for e in self.syms if name in e]
        for e in syms_to_remove:
            self.syms.remove(e)
        del self.nodes[name]

    def rename_node(self, old, new):
        if new in self.nodes:
            raise ValueError("Name already exists.")
        if old not in self.nodes:
            raise ValueError("Node not found.")
        node = self.nodes.pop(old)
        node.name = new
        self.nodes[new] = node

        new_edges = set()
        for e in self.edges:
            if old in e:
                other = next(iter(e - {old}))
                new_edges.add(frozenset({other, new}))
            else:
                new_edges.add(e)
        self.edges = new_edges

    def add_edge(self, name1, name2):
        if name1 == name2:
            return False
        key = frozenset({name1, name2})
        if key in self.edges:
            return False
        if name1 in self.nodes and name2 in self.nodes:
            self.edges.add(key)
            return True
        return False

    def add_sym(self, name1, name2):
        if name1 == name2:
            return False
        if name1 not in self.nodes or name2 not in self.nodes:
            return False
        for syms in self.syms:
            if name1 in syms or name2 in syms:
                return False
        key = frozenset({name1, name2})
        self.syms.add(key)
        return True

    def remove_edge(self, name1, name2):
        key = frozenset({name1, name2})
        if key in self.edges:
            self.edges.remove(key)

    def remove_sym(self, name1, name2):
        key = frozenset({name1, name2})
        if key in self.syms:
            self.syms.remove(key)

    def save_to_yaml(self, filepath): # TODO
        import yaml
        data = {"nodes": [], "connections": [], "symmetry": []}
        node_names = []
        for name, node in self.nodes.items():
            node_names.append(name)
            node_data = {
                "name": name,
                "shape": node.shape,
                "color": node.color.name(),           # QColor -> "#RRGGBB" 
                "line_thickness": node.thickness,
                "filled": node.filled,
                "position": [node.x, node.y]
            }
            if node.shape == 'text':
                node_data["text"] = node.text if node.text is not None else node.name
            data["nodes"].append(node_data)
        if len(node_names) != len(set(node_names)):
            raise ValueError("Duplicate node names found. Cannot save YAML.")
        for edge in self.edges:
            if len(edge) == 2:
                n1, n2 = list(edge)
                data["connections"].append([n1, n2])
        for sym in self.syms:
            if len(sym) == 2:
                n1, n2 = list(sym)
                data["symmetry"].append([n1, n2])
        with open(filepath, 'w') as f:
            yaml.safe_dump(data, f)

    def load_from_yaml(self, filepath):
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        self.nodes.clear()
        self.edges.clear()
        self.syms.clear()
        self.node_counter = 0
        for node_data in data.get("nodes", []):
            name = node_data["name"]
            if name in self.nodes:
                continue
            shape = node_data.get("shape", "circle")
            color = node_data.get("color", "#ffffff")
            thickness = node_data.get("line_thickness", 1)
            filled = node_data.get("filled", False)
            x, y = node_data.get("position", [0, 0])
            text = node_data.get("text") if shape == 'text' else None
            node = Node(name, shape, color, thickness, filled, x, y, text)
            self.nodes[name] = node
            if name.startswith("Node"):
                try:
                    idx = int(name[4:])
                    self.node_counter = max(self.node_counter, idx)
                except:
                    pass
        for conn in data.get("connections", []):
            if len(conn) == 2:
                n1, n2 = conn[0], conn[1]
                if n1 in self.nodes and n2 in self.nodes:
                    self.edges.add(frozenset({n1, n2}))
        for conn in data.get("symmetry", []):
            if len(conn) == 2:
                n1, n2 = conn[0], conn[1]
                if n1 in self.nodes and n2 in self.nodes:
                    self.syms.add(frozenset({n1, n2}))

    def create_training_config(self):
        kpt_names    = list(self.nodes.keys())
        idx_map  = { name: idx for idx, name in enumerate(kpt_names) }
        kpt_perm     = list(range(len(kpt_names)))

        for pair in self.syms:
            n1, n2 = tuple(pair)
            i1, i2 = idx_map[n1], idx_map[n2]
            kpt_perm[i1], kpt_perm[i2] = kpt_perm[i2], kpt_perm[i1]

        return len(kpt_names), kpt_perm, kpt_names