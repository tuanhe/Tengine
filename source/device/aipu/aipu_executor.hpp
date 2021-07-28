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

class AIPUEngine
{
public:
    AIPUEngine();
    ~AIPUEngine() = default;

    int AIPUEnginePreRun(struct subgraph* subgraph);
    int AIPUEngineRun(struct subgraph* subgraph);
    void AIPUEnginePostRun();

private:
    int Build(struct subgraph* subgraph);
    int VXTensorMap(struct graph* ir_graph, int ir_tensor_idx, int spec_type);

    bool AddConvolutionNode(struct node* ir_node);
    bool AddDeconvNode(struct node* ir_node);

private:
    std::shared_ptr<aipubt::Graph> graph;

};
