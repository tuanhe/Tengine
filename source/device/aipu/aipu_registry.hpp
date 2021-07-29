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

#pragma once
#include <functional>
#include <map>

extern "C" {
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "operator/op.h"
#include "utility/log.h"
}

using layer_function_t = std::function<bool(struct node* ir_node)>;
using op_type = uint16_t;

class AIPULayer
{
    public:   
        layer_function_t get_layer_function_by_type(op_type);
        void register_layer(op_type, layer_function_t);  
        static AIPULayer& get_instance();  
    
    private:  
        AIPULayer(){};
        ~AIPULayer();
    
    private:  
        std::map<op_type, layer_function_t> layer_map;  
};

class AIPULayerRegistery {
    public:
        AIPULayerRegistery(op_type op_type,  layer_function_t fn)
        {
            AIPULayer::get_instance().register_layer(op_type, fn);
        }
};

#define AIPU_LAYER_REGISTRY(OPTYPE)          \
       static AIPULayerRegistery g_register_##OPTYPE(OPTYPE, add_##OPTYPE##_node);

