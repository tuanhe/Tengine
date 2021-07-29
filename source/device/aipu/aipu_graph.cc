/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Author: tuanhe
 */

#include "aipu_graph.hpp"
#include "aipu_executor.hpp"
#include "aipu_registry.hpp"

extern "C"
{
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "utility/utils.h"
}


int aipu_dev_init(struct device* dev)
{
    (void)dev;
    //set aipu target
    return 0;
}


int aipu_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options)
{
    //struct subgraph* subgraph= (struct subgraph*)graph ;
    struct graph* ir_graph = subgraph->graph;

    for (size_t i = 0; i < ir_graph->input_num ; ++i )
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, subgraph->input_tensor_list[i]);
        printf("%s  %d  input tensort name  :%s \n", __FUNCTION__, __LINE__,  ir_tensor->name);
        
        //exec_graph->input.tensors[i] = construct_vsi_graph_io_tensor(exec_graph, ir_graph, ir_tensor, tensor_map_list);
    }
    
    for (size_t i = 0; i < ir_graph->output_num; ++i )
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, subgraph->output_tensor_list[i]);

        printf("%s  %d  output tensort name  :%s \n", __FUNCTION__, __LINE__,  ir_tensor->name);
        //exec_graph->output.tensors[i] = construct_vsi_graph_io_tensor(exec_graph, ir_graph, ir_tensor, tensor_map_list);
    }
    
    dump_ir_graph(ir_graph);
    /********** add node ******/
    for(int i = 0 ; i < subgraph->node_num ; i++)
    {
        struct node* ir_node = get_ir_graph_node(ir_graph, subgraph->node_list[i]);

        if(ir_node->op.type == OP_CONST || ir_node->op.type == OP_INPUT)
            continue;
        
        printf("%s  %d  ir node :%s (%s)\n", __FUNCTION__, __LINE__, get_op_name_from_type(ir_node->op.type), ir_node->name);

        auto layer_function = AIPULayer::get_instance().get_layer_function_by_type(ir_node->op.type);
        if( nullptr == layer_function)
        {
            TLOG_ERR("Tengine Fatal: %s is not implemented in the AIPU\n", get_op_name_from_type(ir_node->op.type));
            return -1;
        }

        if(!layer_function(ir_node)){
            TLOG_ERR("Tengine Fatal: node %s parameter filling error\n", ir_node->name);
            return -1;
        }
    }
    
    subgraph->device_graph = new AIPUEngine;
    auto engine = (AIPUEngine*)subgraph->device_graph;

    return engine->AIPUEnginePreRun(subgraph);
}


int aipu_dev_run(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (AIPUEngine*)subgraph->device_graph;
    return engine->AIPUEngineRun(subgraph);
}


int aipu_dev_postrun(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (AIPUEngine*)subgraph->device_graph;
    engine->AIPUEnginePostRun();
    delete engine;

    return 0;
}


int aipu_dev_release(struct device* dev)
{
    (void)dev;
    return 0;
}
