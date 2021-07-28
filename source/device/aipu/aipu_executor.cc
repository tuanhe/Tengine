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

#include "aipu_executor.hpp"
#include "aipu_define.h"

#ifdef TIMVX_MODEL_CACHE
#include "defines.h"
#include "cstdlib"
#endif

#ifdef TIMVX_MODEL_CACHE
#include "tim/vx/ops/nbg.h"
#include <fstream>
#endif


VXEngine::VXEngine()
{
};


int VXEngine::VXTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type)
{
    return 0;
}

int VXEngine::Build(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    for (int i = 0; i < subgraph->node_num; i++)
    {
        uint16_t node_id = subgraph->node_list[i];
        struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
        auto op_type = ir_node->op.type;

        switch (op_type)
        {
            default:
                fprintf(stderr, "Tengine TIM-VX: Cannot support OP(%d).\n", ir_node->index);
                break;
        }
    }

    return 0;
}


int VXEngine::VXEnginePreRun(struct subgraph* subgraph)
{
    struct graph* ir_graph = subgraph->graph;

    return 0;
};

int VXEngine::VXEngineRun(struct subgraph* subgraph)
{
    return 0;
}

void VXEngine::VXEnginePostRun()
{

};
