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
 * Copyright (c) 2021, Open AI Lab
 * Author: lswang@openailab.com
 */

#pragma once

extern "C" {
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"
#include "utility/log.h"
}

#include "gcompiler_api.h"

#define SPEC_TYPE_CONV      1
#define SPEC_TYPE_CONV_BIAS 2
#define SPEC_TYPE_DWCONV    3
#define SPEC_TYPE_INTERP    4
#define SPEC_TYPE_OUTPUT    5
#define SPEC_TYPE_PRELU     6
#define SPEC_TYPE_SLICE     7
#define SPEC_TYPE_RESHAPE   8
#define SPEC_TYPE_INPUT     9

//typedef std::map<uint32_t, std::shared_ptr<tim::vx::Tensor> > dict_irt2vxt;
//typedef std::map<uint32_t, std::shared_ptr<tim::vx::Operation> > dict_irt2vxo;

class VXEngine
{
public:
    VXEngine();
    ~VXEngine() = default;

    int VXEnginePreRun(struct subgraph* subgraph);
    int VXEngineRun(struct subgraph* subgraph);
    void VXEnginePostRun();

private:
    int Build(struct subgraph* subgraph);
    int VXTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type);

    bool AddConvolutionNode(struct node* ir_node);
    bool AddDeconvNode(struct node* ir_node);

public:
    //std::shared_ptr<tim::vx::Context> context;
    //std::shared_ptr<tim::vx::Graph> graph;
    //std::shared_ptr<tim::vx::Operation> ops;
    std::vector<char> nbg_buffer;

private:
    //dict_irt2vxt vx_tensor_map;
    //dict_irt2vxo vx_node_map;
};
