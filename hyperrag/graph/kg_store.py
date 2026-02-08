from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from hyperrag.models.schemas import Entity, Relation


class KnowledgeGraphStore:
    """NetworkX-based knowledge graph for entity/relation storage and querying."""

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_entities(self, entities: list[Entity]) -> None:
        for ent in entities:
            self._graph.add_node(
                ent.name,
                entity_type=ent.entity_type,
                source_doc_id=ent.source_doc_id,
                source_page=ent.source_page,
                source_bbox=ent.source_bbox.model_dump() if ent.source_bbox else None,
            )

    def add_relations(self, relations: list[Relation]) -> None:
        for rel in relations:
            self._graph.add_edge(
                rel.source_entity,
                rel.target_entity,
                relation_type=rel.relation_type,
                source_doc_id=rel.source_doc_id,
                source_page=rel.source_page,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def query_entity(self, entity_name: str) -> dict | None:
        if entity_name not in self._graph:
            return None
        attrs = dict(self._graph.nodes[entity_name])
        neighbors = []
        for _, target, data in self._graph.edges(entity_name, data=True):
            neighbors.append({"target": target, **data})
        for source, _, data in self._graph.in_edges(entity_name, data=True):
            neighbors.append({"source": source, **data})
        return {"name": entity_name, "attributes": attrs, "relations": neighbors}

    def query_neighbors(self, entity_name: str, depth: int = 2) -> dict:
        if entity_name not in self._graph:
            return {"nodes": [], "edges": []}
        nodes_set = {entity_name}
        frontier = {entity_name}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for _, target in self._graph.edges(node):
                    next_frontier.add(target)
                for source, _ in self._graph.in_edges(node):
                    next_frontier.add(source)
            nodes_set |= next_frontier
            frontier = next_frontier

        subgraph = self._graph.subgraph(nodes_set)
        nodes = [
            {"name": n, **dict(subgraph.nodes[n])} for n in subgraph.nodes
        ]
        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in subgraph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def query_by_type(self, entity_type: str) -> list[dict]:
        result = []
        for node, attrs in self._graph.nodes(data=True):
            if attrs.get("entity_type", "").lower() == entity_type.lower():
                result.append({"name": node, **attrs})
        return result

    def get_path(self, source: str, target: str) -> list[str] | None:
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ------------------------------------------------------------------
    # Graph info
    # ------------------------------------------------------------------
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def all_entity_types(self) -> list[str]:
        types = set()
        for _, attrs in self._graph.nodes(data=True):
            if "entity_type" in attrs:
                types.add(attrs["entity_type"])
        return sorted(types)

    def all_nodes(self) -> list[dict]:
        return [
            {"name": n, **dict(attrs)}
            for n, attrs in self._graph.nodes(data=True)
        ]

    def all_edges(self) -> list[dict]:
        return [
            {"source": u, "target": v, **d}
            for u, v, d in self._graph.edges(data=True)
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = nx.node_link_data(self._graph)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._graph = nx.node_link_graph(data, directed=True)
