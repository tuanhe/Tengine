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

#include "aipu_registry.hpp"

AIPULayer::~AIPULayer(){
    layer_map.clear();
};

layer_function_t AIPULayer::get_layer_function_by_type(op_type op_type)
{
    auto iter = layer_map.find(op_type);  
    if(iter == layer_map.end())  
        return nullptr;  
    else  
        return iter->second; 
}

void AIPULayer::register_layer(op_type type, layer_function_t fn)
{
    layer_map.insert(std::pair<op_type, layer_function_t>(type, fn));
}

AIPULayer& AIPULayer::get_instance()
{
    static AIPULayer aipu_layers;  
    return aipu_layers; 
}
    

