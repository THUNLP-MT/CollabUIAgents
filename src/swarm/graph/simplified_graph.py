import random
from typing import Optional
import asyncio
from swarm.graph.graph import Graph
from swarm.graph.node import Node
from swarm.environment.agents import AgentRegistry
#print('registry:', AgentRegistry.registry.keys())


class SimpleGraph(Graph):
    def __init__(self, 
                domain: str,
                model_name: Optional[str] = None,
                meta_prompt: bool = False,
                ):
        super().__init__(domain, model_name)
        self.domain = domain
        self.model_name = model_name
        self.graph = Graph(domain, model_name)

    def graph_organize(self, agent_list, connection_pair):
        idx_id = {}
        agent_instances = []
        for idx, agent_info in enumerate(agent_list):
            if agent_info[0] in AgentRegistry.registry:
                #print('agent_info2:', agent_info[0])
                agent_instance = AgentRegistry.get(agent_info[0], model_name=self.model_name, temperature=agent_info[1])
                agent_instances.append(agent_instance)
                node_id = self.graph.add_node(agent_instance).id
                idx_id[idx] = node_id
        print('idx_id:', idx_id)

        for connection in connection_pair: # node: {id: Node}
            idx_predecessor = connection[0]
            idx_successor = connection[1]
            id_predecessor = idx_id[connection[0]]
            id_successor = idx_id[connection[1]]

            self.graph.nodes[id_predecessor].add_successor(self.graph.nodes[id_successor])

        self.graph.input_nodes = agent_instances
        self.graph.output_nodes = agent_instances

        #return self.graph

    
    async def evaluate(self, inputs):
        answer = await self.graph.run(inputs, max_time=10000, max_tries=1, return_all_outputs=True)
        return answer


