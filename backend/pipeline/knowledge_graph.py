"""Knowledge Graph Construction for Video Content

Builds NetworkX graphs from transcript chunks using:
- Entity extraction (noun phrases, named entities)
- Relation extraction (subject-verb-object triples)
- Temporal anchoring (timestamps)
- Graph analytics (centrality, clustering)
"""

import logging
import pickle
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
from collections import defaultdict, Counter

from pipeline.spacy_util import load_spacy_model

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Knowledge graph builder for video transcript analysis"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize knowledge graph builder
        
        Args:
            spacy_model: Name of spaCy model to use
        """
        self.spacy_model = spacy_model
        self.nlp = None
        self.graph = None
        
    def initialize(self):
        """Load spaCy model"""
        try:
            self.nlp = load_spacy_model(self.spacy_model)
            logger.info(f"Loaded spaCy model: {self.spacy_model}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise
    
    def build_knowledge_graph(self,
                             chunks: List[Dict[str, Any]],
                             video_id: str,
                             progress_callback: Optional[callable] = None) -> nx.DiGraph:
        """
        Build knowledge graph from transcript chunks using batched spaCy pipe.
        """
        if not self.nlp:
            self.initialize()

        if progress_callback:
            progress_callback("Initializing knowledge graph...", 10)

        self.graph = nx.DiGraph(video_id=video_id)

        if progress_callback:
            progress_callback("Extracting entities and relations (batched)...", 30)

        # Batch process all chunk texts in a single pipe pass — much faster than
        # calling self.nlp(text) per chunk (avoids Python overhead per call).
        texts = [c["text"] for c in chunks]
        all_entities: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []

        n = max(1, len(chunks))
        for i, (chunk, doc) in enumerate(
            zip(chunks, self.nlp.pipe(texts, batch_size=16))
        ):
            if progress_callback and i % 20 == 0:
                progress_callback(
                    f"NLP {i+1}/{n}", 30 + (i * 40 // n)
                )
            entities, relations = self._extract_from_doc(doc, chunk)
            all_entities.extend(entities)
            all_relations.extend(relations)

        if progress_callback:
            progress_callback("Building graph structure...", 75)

        self._add_entities_to_graph(all_entities)
        self._add_relations_to_graph(all_relations)
        self._add_cooccurrence_edges(all_entities)

        if progress_callback:
            progress_callback("Computing graph analytics...", 92)

        # Cheap metrics only (degree centrality is O(V+E)). Skip betweenness/closeness:
        # both are O(V*E) and weren't actually consumed by retrieval.
        self._compute_graph_metrics()

        if progress_callback:
            progress_callback(
                f"KG complete: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges",
                100,
            )

        return self.graph

    # Pronouns / determiners that should never become standalone graph entities
    _STOP_NOUN_CHUNKS = frozenset({
        "i", "you", "we", "they", "it", "he", "she", "this", "that",
        "these", "those", "what", "which", "who", "whom", "us", "them",
        "me", "him", "her", "myself", "yourself", "themselves", "everyone",
        "someone", "anything", "everything", "something", "nothing",
        "one", "ones", "thing", "things", "way", "ways", "kind", "kinds",
    })

    def _extract_from_doc(self, doc, chunk: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities + relations from an already-parsed spaCy doc."""
        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []
        
        # Extract entities
        # 1. Noun phrases as concepts (filter pronouns / determiners / single short words)
        for noun_chunk in doc.noun_chunks:
            text = noun_chunk.text.lower().strip()
            words = text.split()
            if not text or len(words) > 4 or len(text) < 3:
                continue
            if text in self._STOP_NOUN_CHUNKS:
                continue
            # Skip if the head is a pronoun
            if noun_chunk.root.pos_ in {"PRON", "DET"}:
                continue
            # Strip determiner if present (e.g. "the dataframe" → "dataframe")
            if words[0] in {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}:
                text = " ".join(words[1:]).strip()
                if not text or len(text) < 3 or text in self._STOP_NOUN_CHUNKS:
                    continue
            entities.append({
                "text": text,
                "label": "CONCEPT",
                "chunk_id": chunk["chunk_id"],
                "time_start": chunk["time_start"],
                "time_end": chunk["time_end"],
                "confidence": self._compute_entity_confidence(noun_chunk),
            })
        
        # 2. Named entities
        for ent in doc.ents:
            entities.append({
                "text": ent.text.lower().strip(),
                "label": ent.label_,
                "chunk_id": chunk["chunk_id"],
                "time_start": chunk["time_start"],
                "time_end": chunk["time_end"],
                "confidence": self._compute_entity_confidence(ent)
            })
        
        # Extract relations (subject-verb-object triples)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Find subjects
                subjects = [child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}]
                # Find objects
                objects = [child for child in token.children if child.dep_ in {"dobj", "pobj", "attr"}]
                
                for subj in subjects:
                    for obj in objects:
                        relations.append({
                            "subject": subj.text.lower().strip(),
                            "predicate": token.lemma_.lower().strip(),
                            "object": obj.text.lower().strip(),
                            "chunk_id": chunk["chunk_id"],
                            "time_start": chunk["time_start"],
                            "time_end": chunk["time_end"],
                            "confidence": self._compute_relation_confidence(subj, token, obj)
                        })
        
        return entities, relations
    
    def _compute_entity_confidence(self, entity) -> float:
        """Compute confidence score for an entity"""
        # Base confidence depends on entity type
        if hasattr(entity, 'label_'):
            if entity.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                base_confidence = 0.9
            elif entity.label_ in ["DATE", "TIME", "MONEY", "QUANTITY"]:
                base_confidence = 0.85
            else:
                base_confidence = 0.7
        else:
            # For noun chunks
            base_confidence = 0.6
        
        # Adjust based on length and complexity
        text = entity.text.lower().strip()
        if len(text.split()) == 1:
            base_confidence *= 0.9  # Single words might be less specific
        elif len(text.split()) > 3:
            base_confidence *= 0.8  # Very long phrases might be noisy
        
        return min(base_confidence, 1.0)
    
    def _compute_relation_confidence(self, subj: Any, verb: Any, obj: Any) -> float:
        """Compute confidence score for a relation"""
        base_confidence = 0.7
        
        # Boost confidence for clear grammatical structures
        if subj.dep_ == "nsubj" and obj.dep_ == "dobj":
            base_confidence = 0.85
        elif subj.dep_ == "nsubjpass" and obj.dep_ in {"pobj", "attr"}:
            base_confidence = 0.8
        
        # Adjust based on verb specificity
        if verb.lemma_ in ["be", "have", "do"]:
            base_confidence *= 0.8  # Common verbs might be less informative
        
        return min(base_confidence, 1.0)
    
    def _add_entities_to_graph(self, entities: List[Dict[str, Any]]):
        """Add entity nodes to the graph"""
        for entity in entities:
            node_id = entity["text"].lower().strip()
            
            if self.graph.has_node(node_id):
                # Update existing node
                node_data = self.graph.nodes[node_id]
                node_data["mentions"].append(entity["time_start"])
                node_data["chunk_ids"].add(entity["chunk_id"])
                node_data["confidence"] = max(node_data["confidence"], entity["confidence"])
            else:
                # Add new node
                self.graph.add_node(
                    node_id,
                    label=entity["label"],
                    mentions=[entity["time_start"]],
                    chunk_ids={entity["chunk_id"]},
                    time_start=entity["time_start"],
                    time_end=entity["time_end"],
                    confidence=entity["confidence"],
                    entity_type=entity["label"]
                )
    
    def _add_relations_to_graph(self, relations: List[Dict[str, Any]]):
        """Add relation edges to the graph"""
        for relation in relations:
            src = relation["subject"].lower().strip()
            dst = relation["object"].lower().strip()
            
            # Ensure nodes exist
            if not self.graph.has_node(src):
                self.graph.add_node(
                    src,
                    label="MISC",
                    mentions=[],
                    chunk_ids=set(),
                    time_start=relation["time_start"],
                    time_end=relation["time_end"],
                    confidence=0.5,
                    entity_type="MISC"
                )
            
            if not self.graph.has_node(dst):
                self.graph.add_node(
                    dst,
                    label="MISC",
                    mentions=[],
                    chunk_ids=set(),
                    time_start=relation["time_start"],
                    time_end=relation["time_end"],
                    confidence=0.5,
                    entity_type="MISC"
                )
            
            # Add or update edge
            if self.graph.has_edge(src, dst):
                edge_data = self.graph[src][dst]
                edge_data["predicates"].append(relation["predicate"])
                edge_data["chunk_ids"].add(relation["chunk_id"])
                edge_data["confidence"] = max(edge_data["confidence"], relation["confidence"])
            else:
                self.graph.add_edge(
                    src, dst,
                    predicates=[relation["predicate"]],
                    chunk_ids={relation["chunk_id"]},
                    time_start=relation["time_start"],
                    time_end=relation["time_end"],
                    confidence=relation["confidence"],
                    relation_type="semantic"
                )
    
    def reset(self):
        """Clear the graph so the next task starts with a clean slate."""
        self.graph = None

    def _add_cooccurrence_edges(self, entities: List[Dict[str, Any]]):
        """Add co-occurrence edges between entities that appear in the same chunk."""
        chunk_entity_map: Dict[str, List[str]] = defaultdict(list)
        for entity in entities:
            chunk_entity_map[entity["chunk_id"]].append(entity["text"])

        for chunk_id, entity_list in chunk_entity_map.items():
            seen = list(dict.fromkeys(entity_list))  # deduplicate, preserve order
            for i, ent_a in enumerate(seen):
                for ent_b in seen[i + 1:]:
                    if not self.graph.has_edge(ent_a, ent_b):
                        self.graph.add_edge(
                            ent_a, ent_b,
                            predicates=["co-occurs"],
                            chunk_ids={chunk_id},
                            time_start=0,
                            time_end=0,
                            confidence=0.5,
                            relation_type="co-occurrence",
                            weight=1,
                        )
                    else:
                        self.graph[ent_a][ent_b].get("chunk_ids", set()).add(chunk_id)
                        self.graph[ent_a][ent_b]["weight"] = (
                            self.graph[ent_a][ent_b].get("weight", 1) + 1
                        )

    def _compute_graph_metrics(self):
        """Compute cheap graph analytics. Heavy O(V*E) metrics intentionally skipped:
        retrieval doesn't consume them and on a 100-node graph they cost ~seconds.
        """
        if not self.graph:
            return

        # Degree centrality is O(V+E)
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            for node in self.graph.nodes():
                self.graph.nodes[node]["degree_centrality"] = degree_centrality.get(node, 0.0)
        except Exception as e:
            logger.warning(f"Failed to compute degree centrality: {e}")

        # Density is cheap; skip clustering coefficient (requires undirected conversion + N(v) iteration)
        try:
            self.graph.graph["density"] = nx.density(self.graph)
        except Exception as e:
            logger.warning(f"Failed to compute graph density: {e}")
    
    def search_entities(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for entities matching the query"""
        if not self.graph:
            return []
        
        query_lower = query.lower()
        results = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # Bidirectional substring match (matches notebook formula)
            node_lower = node.lower()
            match_score = 0.0
            if query_lower in node_lower or node_lower in query_lower:
                match_score = 1.0
            
            if match_score > 0:
                results.append({
                    "entity": node,
                    "match_score": match_score,
                    "confidence": node_data.get("confidence", 0.0),
                    "mentions": node_data.get("mentions", []),
                    "entity_type": node_data.get("entity_type", ""),
                    "degree_centrality": node_data.get("degree_centrality", 0.0),
                    "chunk_ids": list(node_data.get("chunk_ids", set()))
                })
        
        # Sort by mention count so high-frequency entities surface first
        results.sort(key=lambda x: len(x["mentions"]), reverse=True)
        return results[:top_k]
    
    def get_related_entities(self, entity: str, relation_type: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get entities related to the given entity"""
        if not self.graph or not self.graph.has_node(entity):
            return []
        
        related = []
        node_data = self.graph.nodes[entity]
        
        # Get outgoing neighbors
        for neighbor in self.graph.successors(entity):
            edge_data = self.graph[entity][neighbor]
            if relation_type is None or relation_type in edge_data.get("predicates", []):
                related.append({
                    "entity": neighbor,
                    "relation": "outgoing",
                    "predicates": edge_data.get("predicates", []),
                    "confidence": edge_data.get("confidence", 0.0),
                    "chunk_ids": list(edge_data.get("chunk_ids", set()))
                })
        
        # Get incoming neighbors
        for neighbor in self.graph.predecessors(entity):
            edge_data = self.graph[neighbor][entity]
            if relation_type is None or relation_type in edge_data.get("predicates", []):
                related.append({
                    "entity": neighbor,
                    "relation": "incoming",
                    "predicates": edge_data.get("predicates", []),
                    "confidence": edge_data.get("confidence", 0.0),
                    "chunk_ids": list(edge_data.get("chunk_ids", set()))
                })
        
        # Sort by confidence
        related.sort(key=lambda x: x["confidence"], reverse=True)
        return related[:top_k]
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the knowledge graph"""
        if not self.graph:
            return {}
        
        # Entity type distribution
        entity_types = Counter()
        for node in self.graph.nodes():
            entity_type = self.graph.nodes[node].get("entity_type", "UNKNOWN")
            entity_types[entity_type] += 1
        
        # Relation type distribution
        relation_types = Counter()
        for src, dst in self.graph.edges():
            edge_data = self.graph[src][dst]
            for predicate in edge_data.get("predicates", []):
                relation_types[predicate] += 1
        
        # Top entities by centrality
        top_entities = sorted(
            [(node, self.graph.nodes[node].get("degree_centrality", 0.0)) 
             for node in self.graph.nodes()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": self.graph.graph.get("density", 0.0),
            "clustering_coefficient": self.graph.graph.get("clustering_coefficient", 0.0),
            "num_connected_components": self.graph.graph.get("num_connected_components", 0),
            "entity_types": dict(entity_types.most_common()),
            "relation_types": dict(relation_types.most_common(10)),
            "top_entities": top_entities
        }
    
    def save_graph(self, output_dir: str, video_id: str):
        """Save knowledge graph to disk."""
        os.makedirs(output_dir, exist_ok=True)

        kg_path = os.path.join(output_dir, f"{video_id}_knowledge_graph.pkl")
        with open(kg_path, "wb") as f:
            pickle.dump(self.graph, f)

        # Edge list and summary are diagnostic-only — failures are non-fatal
        try:
            edges_data = []
            for src, dst in self.graph.edges():
                ed = self.graph[src][dst]
                edges_data.append({
                    "source": str(src),
                    "target": str(dst),
                    "predicates": list(ed.get("predicates", [])),
                    "confidence": float(ed.get("confidence", 0.0)),
                    "chunk_ids": list(ed.get("chunk_ids", set())),
                })
            kg_json_path = os.path.join(output_dir, f"{video_id}_kg_edges.json")
            with open(kg_json_path, "w", encoding="utf-8") as f:
                json.dump(edges_data[:200], f, indent=2)
        except Exception as e:
            logger.warning(f"KG edge-list save failed (non-fatal): {e}")

        try:
            summary = self.get_graph_summary()
            # Ensure top_entities tuples are lists (JSON-safe)
            summary["top_entities"] = [list(t) for t in summary.get("top_entities", [])]
            summary_path = os.path.join(output_dir, f"{video_id}_kg_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"KG summary save failed (non-fatal): {e}")

        logger.info(f"Saved knowledge graph → {kg_path}")
        return kg_path
    
    @classmethod
    def load_graph(cls, graph_path: str) -> nx.DiGraph:
        """Load knowledge graph from disk"""
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        return graph
