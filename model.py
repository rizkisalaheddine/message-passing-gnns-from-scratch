"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
import torch 
import numpy as np 
def edges_to_coo(edge_list, num_nodes=None):
    # TODO: Convert a list of (src, dst) edge pairs into COO-format src/dst tensors.
    src = torch.zeros(len(edge_list),dtype=torch.long)
    dst = torch.zeros(len(edge_list),dtype=torch.long)
    idx = 0

    while idx < len(edge_list) : 
        src[idx] = edge_list[idx][0]
        dst[idx] = edge_list[idx][1]
        idx += 1 
    if num_nodes != None : 
        num_nodes = num_nodes
    elif edge_list == []:
        num_nodes  = 0
    else : 
        num_nodes = max([i for i in src])+1
        num_nodes = num_nodes.item()
    
    return src, dst, num_nodes

# Step 2 - add_self_loops
def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    # TODO: Append self-loop edges (i, i) for every node to the COO tensors
    self_loops = torch.zeros(num_nodes,dtype=src.dtype)
    self_loops.shape
    for i in range(num_nodes):
        self_loops[i] = i
    src = torch.cat((src,self_loops),0)
    dst = torch.cat((dst,self_loops),0)
    return src, dst

# Step 3 - compute_node_degrees
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    # TODO: Compute per-node in-degrees by scattering onto destination nodes
    weights = torch.zeros(num_nodes,dtype=torch.float32)
    ones = torch.ones_like(src,dtype=torch.float32)

    for i in range(num_nodes):
        t = torch.where(dst==i,1.0,0.0)

        if edge_weight != None : 
            w = t.T @ edge_weight
        else : 
            w = t.T @ ones 

        weights[i] = w.item()
        

    return weights

# Step 4 - symmetric_normalize_edge_weights
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    # TODO: Compute symmetrically normalized edge weights for GCN-style propagation.
    normalized_weights = torch.zeros_like(src,dtype=torch.float32)
    edge_weights = edge_weight if edge_weight is not None else torch.ones_like(src,dtype=torch.float32)
    degree = torch.zeros(num_nodes,dtype=torch.float32)

    # compute node degrees
    for i in range(num_nodes):
        t = torch.where(dst==i,1.0,0.0)
        w = t.T @ edge_weights
        degree[i] = w.item()
    
    for i in range(len(src)) : 
        if degree[src[i]] == 0 or degree[dst[i]] == 0 : 
            normalized_weights[i] = 0
        else : 
            normalized_weights[i] = edge_weights[i].item()/(torch.sqrt(degree[src[i]]).item()*torch.sqrt(degree[dst[i]]).item())
    
    return normalized_weights

# Step 5 - gather_source_node_features
def gather_source_node_features(node_features, src):
    # TODO: Return edge-aligned source feature rows (E, F) from node_features.
    n,f  = node_features.shape
    out = torch.zeros(len(src),f)

    for i in range(len(src)) : 
        out[i] = node_features[src[i]]
    
    return out

# Step 6 - scatter_sum_to_nodes
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    # TODO: Scatter-sum edge features onto destination nodes to produce per-node vectors
    e, f = edge_features.shape
    out = torch.zeros((num_nodes,f),dtype=edge_features.dtype)

    for j in range(num_nodes) : 
        indices, = torch.where(dst == j)
        out[j] = edge_features[indices].sum(dim=0)
    
    return out

# Step 7 - scatter_mean_to_nodes
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-mean edge features onto destination nodes (sum then divide by in-degree).
    e, f = edge_features.shape
    out = torch.zeros((num_nodes,f))
    for i in range(num_nodes) : 
        indices, = torch.where(dst == i)
        
        out[i] = edge_features[indices].sum(dim=0)
        if len(indices) !=  0 :
            out[i] *= 1/len(indices) 
    return out

# Step 8 - scatter_max_to_nodes
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-max edge features onto destination nodes (elementwise max).
    e, f =edge_features.shape
    out = torch.full((num_nodes,f),float("-inf"))

    for i in range(num_nodes) : 
        ids = edge_features[torch.where(dst==i)]
        if ids.numel() > 0 :
            out[i] = torch.max(ids,0).values
        else : 
            pass
    return out

# Step 9 - compute_messages
def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    # TODO: Build per-edge messages by gathering features and applying message_fn
    src_ft = gather_source_node_features(node_features,src)
    dst_ft = gather_source_node_features(node_features,dst)
    if edge_attr is not None : 
        messages = message_fn(src_ft,dst_ft,edge_attr)
    else : 
        messages = message_fn(src_ft,dst_ft)
    return messages

# Step 10 - aggregate_messages
def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    # TODO: Aggregate edge messages onto destination nodes via sum/mean/max...
    e, m = messages.shape
    agg_msg = torch.zeros((num_nodes,m),dtype=torch.float32)
    
    if aggr == "sum" : 
        agg_msg = scatter_sum_to_nodes(messages, dst, num_nodes)
    elif aggr == "mean" : 
        agg_msg = scatter_mean_to_nodes(messages, dst, num_nodes)
    elif aggr == "max" : 
        agg_msg = scatter_max_to_nodes(messages, dst, num_nodes)
    else : 
        raise ValueError("Not Supported type of aggregation")   
    
    return agg_msg

# Step 11 - update_node_features
def update_node_features(node_features, aggregated, update_fn):
    # TODO: Implement update_node_features to fuse each node's current state with its aggregated...

    return update_fn(node_features,aggregated)

# Step 12 - message_passing_layer
def message_passing_layer(node_features, src, dst, message_fn, update_fn, aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    # TODO: compose message, aggregate, and update into one MPNN step
    num_nodes,_ = node_features.shape
    messages = compute_messages(node_features, src, dst, message_fn, edge_attr)
    aggs = aggregate_messages(messages, dst, num_nodes, aggr)
    updated_features = update_node_features(node_features, aggs, update_fn)

    return updated_features

# Step 13 - stack_message_passing_layers
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    # TODO: Apply a sequence of MP layer callables; return final + intermediates
    all_layer_outputs = []
    embeddings = node_features
    for layer in layers : 
        embeddings = layer(embeddings, src, dst, edge_attr)
        all_layer_outputs.append(embeddings)
    return embeddings, all_layer_outputs

# Step 14 - gcn_renormalize_adjacency
def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    # TODO: add self-loops then symmetrically normalize the adjacency...
    src_hat, dst_hat = add_self_loops(src, dst, num_nodes)
    norm_weight = symmetric_normalize_edge_weights(src_hat, dst_hat, num_nodes)

    return src_hat, dst_hat, norm_weight

# Step 15 - gcn_linear_transform
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: compute the matrix product and optionally add a bias vector
    out = node_features @ weight 
    if bias is not None : 
        out += bias
    return out

# Step 16 - gcn_layer_forward
def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: Forward pass of one GCN layer: renormalize, transform, propagate...
    
    if num_nodes == None : 
        num_nodes, _ = node_features.shape
    
    src_hat, dst_hat, norm_weight = gcn_renormalize_adjacency(src, dst, num_nodes)
    out = gcn_linear_transform(node_features, weight, bias)
    propagate = torch.zeros_like(out)
    propagate.index_add_(
        0,
        dst_hat,
        torch.multiply(out[src_hat] ,norm_weight.unsqueeze(-1))
    )
     
    if activation is not None : 
        propagate = activation(propagate)
    return propagate

# Step 17 - init_gcn_parameters
def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    # TODO: Initialize GCN weight (and optional bias) with Glorot-style uniform...
    out = {}
    if seed is not None : 
        torch.manual_seed(seed)
    a = np.sqrt(6 / (in_dim + out_dim))
    weight = (2*a)*torch.rand(in_dim, out_dim) - a

    out['weight'] = weight
    
    if with_bias == True : 
        bias = torch.zeros((out_dim,))
        
        out['bias'] = bias

    return     out

# Step 18 - gcn_stack_forward
def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    # TODO: Run a stack of GCN layers to produce deep node embeddings
    all_layer_outputs = []
    embeddings = node_features
    
    if num_nodes == None : 
        num_nodes, _ = node_features.shape

    for i in range (len(param_list)):
            weight = param_list[i]['weight']
            bias = None
            if "bias" in param_list[i].keys() :
                bias = param_list[i]['bias']
            if activations is not None: 
                embeddings = gcn_layer_forward(embeddings, src, dst, weight, bias, num_nodes, activations[i])
            else : 
                embeddings = gcn_layer_forward(embeddings, src, dst, weight, bias, num_nodes)
            all_layer_outputs.append(embeddings)
    return embeddings, all_layer_outputs

# Step 19 - gat_attention_logits
def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    # TODO: return per-edge LeakyReLU attention logits and transformed features
    transformed = node_features @ weight
    att = (
        (transformed[src] * attn_src).sum(dim=-1)
        + (transformed[dst] * attn_dst).sum(dim=-1)
    )
    logits = torch.nn.functional.leaky_relu(att,0.2)
    
    return logits, transformed

# Step 20 - gat_masked_neighbor_softmax
def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    # TODO: Numerically stable softmax of attention logits over each dest node's neighbors
    out = torch.zeros_like(logits)
    softmax = torch.nn.Softmax(dim=0)
    for i in range(num_nodes) :
        idcs = dst == i 
        lgs = logits[idcs]
        #stable softmax by substracting alpha
        if lgs.numel()>0:
            alpha = max(lgs).item()
        else :
            alpha = float("-inf")
        out[idcs] =softmax(lgs-alpha)
        
    return out

# Step 21 - gat_head_forward
def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    # TODO: Forward pass of a single GAT attention head: transform, coeffs, aggregate...
    

    if num_nodes == None : 
        num_nodes, _  = node_features.shape
        
    logits, transformed = gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight)
    head_out = torch.zeros_like(transformed)
    att_coeff = gat_masked_neighbor_softmax(logits, dst, num_nodes)
    for i in range(num_nodes) : 
        head_out[i] = (att_coeff[dst==i].unsqueeze(-1)*transformed[src[dst==i]]).sum(dim=0)
    if bias != None : 
            head_out += bias   
    if activation is not None : 
        head_out = activation(head_out)
    return head_out, att_coeff

# Step 22 - merge_gat_heads
def merge_gat_heads(head_outputs, mode='concat'):
    # TODO: Merge multi-head GAT outputs into one node-feature tensor.
    assert len(head_outputs) > 0 
    if isinstance(head_outputs, (list, tuple)):
        if mode == "concat" :
            return torch.cat(head_outputs,dim=1)
        elif mode == "mean" :
            hs=torch.stack(head_outputs,0)
            return torch.mean(hs,dim=0)
        else : 
            raise ValueError("Not supported mode")
    else : 
        h,n,f= head_outputs.shape
        if mode == "concat" :
            return head_outputs.permute(1, 0, 2).reshape(n,-1)
        elif mode == "mean" :
            return head_outputs.mean(dim=0)
        else : 
            raise ValueError("Not supported mode")

# Step 23 - gat_layer_forward
def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    # TODO: run each head, merge outputs, apply optional nonlinearity...
    if num_nodes == None : 
        num_nodes,_ = node_features.shape
    all_attn = []
    head_outputs = []
    for i in range(len(head_params)) :
        weight = head_params[i]["weight"]
        attn_src = head_params[i]["attn_src"]
        attn_dst =  head_params[i]["attn_dst"]
        bias = head_params[i].get("bias",None)
               
        head_out, atts = gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias, num_nodes)
        all_attn.append(atts)
        head_outputs.append(head_out)
    out = merge_gat_heads(head_outputs, merge_mode) 
    if activation is not None : 
        out = activation(out)
    return out, all_attn

# Step 24 - init_gat_parameters
def init_gat_parameters(in_dim, out_dim, num_heads=1, with_bias=True, seed=None):
    # TODO: Initialize multi-head GAT parameters with Glorot-style initialization.
    if seed is not None : 
        torch.manual_seed(seed)
    a_w = np.sqrt(6/(in_dim+out_dim))
    a_a = np.sqrt(6/(out_dim+1))
    out = []
    
    for i in range(num_heads) : 
        param = {}
        weight = (2*a_w) * torch.rand((in_dim,out_dim),requires_grad=True) - a_w
        attn_src = (2*a_a) * torch.rand((out_dim,),requires_grad=True) - a_a
        attn_dst = (2*a_a) * torch.rand((out_dim,),requires_grad=True) - a_a
        if with_bias is True : 
            bias = torch.zeros((out_dim,),requires_grad=True)
            param["bias"] = bias
        param["weight"] = weight
        param["attn_src"] = attn_src
        param["attn_dst"] = attn_dst
        out.append(param)
    return out

# Step 25 - gat_stack_forward
def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    # TODO: Run a stack of multi-head GAT layers for deep node embeddings.
    if num_nodes is None : 
        num_nodes, _ = node_features.shape
    all_layer_outputs = []
    num_layers = len(layer_param_list)
    if merge_modes is None:
        merge_modes = ["concat"] * num_layers

    if activations is None:
        activations = [None] * num_layers
    for l in range(num_layers):
            layer_output, _ = gat_layer_forward(node_features, src, dst, layer_param_list[l], merge_mode=merge_modes[l], num_nodes=num_nodes, activation=activations[l])
            all_layer_outputs.append(layer_output)
        
    return all_layer_outputs[-1],  all_layer_outputs

# Step 26 - global_mean_pool
def global_mean_pool(node_features, batch_index, num_graphs=None):
    """Globally mean-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one feature row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id in
            {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, F); row b is the mean of node features with
        batch_index == b.
    """
    # TODO: Mean-pool node features into one graph-level vector per graph...
    n, f = node_features.shape
    if num_graphs is None : 
        num_graphs  = batch_index.max() + 1
    out = torch.zeros((num_graphs,f))
    for b in range(num_graphs) : 
        indices = batch_index == b
        out[b] = torch.mean(node_features[indices],dim=0)
    return out

# Step 27 - global_sum_pool
def global_sum_pool(node_features, batch_index, num_graphs=None):
    """Globally sum-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id
            in 0 .. B-1.
        num_graphs: optional int B. If None, inferred as max(batch_index) + 1.

    Returns:
        FloatTensor of shape (B, F); row g is the sum of node features with
        batch_index == g.
    """
    # TODO: sum-pool node features into one graph-level vector per graph
    n, f = node_features.shape
    if num_graphs is None : 
        num_graphs  = batch_index.max() + 1
    out = torch.zeros((num_graphs,f))
    for b in range(num_graphs) : 
        indices = batch_index == b
        out[b] = torch.sum(node_features[indices],dim=0)
    return out

# Step 28 - global_max_pool
def global_max_pool(node_features, batch_index, num_graphs=None):
    # TODO: Globally max-pool node features into one graph-level vector per graph.
    n, f = node_features.shape
    if num_graphs is None : 
        num_graphs  = batch_index.max() + 1
    out = torch.full((num_graphs,f),float("-inf"),dtype=node_features.dtype,device=node_features.device)
    for b in range(num_graphs) : 
        indices = batch_index == b
        if indices.any() : 
            out[b] = torch.max(node_features[indices],dim=0).values
        
    return out

# Step 29 - global_mean_max_pool
def global_mean_max_pool(node_features, batch_index, num_graphs=None):
    """Concatenate global mean and max pooled features into a 2F-dim graph vector.

    Args:
        node_features: FloatTensor of shape (N, F).
        batch_index: LongTensor of shape (N,) with graph ids in {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, 2F); each row is [mean_pool || max_pool].
    """
    # TODO: Concatenate global mean and max pooled features into a 2F-dim vector...
    global_mean = global_mean_pool(node_features, batch_index, num_graphs)
    global_max = global_max_pool(node_features, batch_index, num_graphs)

    return torch.cat((global_mean,global_max),dim=1)

# Step 30 - node_classification_head
def node_classification_head(node_embeddings, weight, bias=None):
    # TODO: Map node embeddings to per-node class logits via a linear head...
    out = node_embeddings @ weight
    if bias is not None : 
        out += bias
    return out

# Step 31 - graph_regression_head
def graph_regression_head(graph_embeddings, weight, bias=None):
    # TODO: Map pooled graph embeddings to regression predictions via a linear head.
    out = graph_embeddings @ weight.T
    if bias is not None : 
        out += bias
    return out

# Step 32 - generate_sbm_graph
def generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Sample one SBM graph with community labels and random node features.
    if seed is not None : 
        torch.manual_seed(seed)

    labels = torch.zeros(num_nodes,dtype=torch.long)
    out = {}
    for c in range(num_classes) : 
        start = c * num_nodes // num_classes
        end = (c+1) * num_nodes // num_classes

        labels[start:end] = c
    
    src = []
    dst = []
 
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  
            if labels[i] == labels[j]:
                p = p_in
            else:
                p = p_out

            if torch.rand(1).item() < p:
                
                src.extend([i, j])
                dst.extend([j, i])


    if len(src) > 0:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    node_features = torch.randn(
        (num_nodes, feature_dim),
        dtype=torch.float32
    )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "node_labels": labels,
        "num_nodes": num_nodes
    }

# Step 33 - build_node_classification_dataset
def build_node_classification_dataset(num_graphs, num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Build a list of SBM graphs with consistent schema for node classification.
    l=[]
    for g in range(num_graphs) : 
        if seed is not None : 
            g_seed = seed * (1+g) # different seed for each graph
            torch.manual_seed(g_seed)
        l.append(generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=g_seed))
    return l

# Step 34 - generate_molecule_like_graph
def generate_molecule_like_graph(num_nodes, num_node_features, edge_prob=0.3, seed=0):
    # TODO: Synthesize one molecule-like graph with features, edges, and target...
    torch.manual_seed(seed)
    out = {}
    x = torch.randn((num_nodes,num_node_features),dtype=torch.float32)
    nodes = torch.tensor([i for i in range(num_nodes)],dtype=torch.long)

    pairs = torch.combinations(nodes,r=2)

    pairs = torch.permute(pairs,(1,0))

    
    probs = torch.tensor([torch.rand(1).item() < edge_prob for _ in range(pairs.shape[1])],dtype=torch.bool)

    edge_index_1  = pairs[:,probs]
    edge_index_2 = edge_index_1[[1,0]]
    edge_index = torch.cat((edge_index_1,edge_index_2),dim=1)
    degrees = compute_node_degrees(edge_index[0], edge_index[1], num_nodes, edge_weight=None)

    y = 0
    y= torch.tensor(y,dtype=torch.float32)
    for v in range(num_nodes) : 
        deg_v = degrees[v]
        a = x[v].mean()
        y += deg_v * a
    
    y = y/num_nodes

    out["x"] = x
    out["edge_index"] = edge_index
    out["y"] = y 

    return out

# Step 35 - build_graph_regression_dataset
def build_graph_regression_dataset(num_graphs, num_nodes_range, num_node_features, edge_prob=0.3, seed=0):
    # TODO: Build a list of molecule-like graphs for graph-level regression.
    out = []
    for g in range(num_graphs) : 
        torch.manual_seed(seed+g)
        lo, hi = num_nodes_range
        num_nodes = lo + (g % (hi - lo + 1))
        out.append(generate_molecule_like_graph(num_nodes, num_node_features, edge_prob, seed=seed+g))
    return out

# Step 36 - collate_graph_batch
def collate_graph_batch(graphs):
    # TODO: Combine variable-size graphs into one disconnected batched graph.
    edge_index = []
    batch = []
    x = []
    y = []
    out = {}
    node_offset = 0
    for g in range(len(graphs)) : 
        if isinstance(graphs[g]["y"],float) : 
            y_g = torch.tensor(graphs[g]["y"],dtype=torch.float32)
        else :
            y_g = graphs[g]["y"]
        x_g = graphs[g]["x"]
        edge_index_g = graphs[g]["edge_index"] + node_offset

        node_offset+=x_g.shape[0]
        batch.append(torch.full((x_g.shape[0],),g,dtype=torch.long))
        x.append(x_g)
        y.append(y_g)
        edge_index.append(edge_index_g)
    
    out["edge_index"] = torch.cat([x for x in edge_index],1) 
    out["batch"] = torch.cat([x for x in batch])
    out["x"] =  torch.cat([f for f in x],dim=0)
    out["y"] = torch.stack([x for x in y])

    
    return out

# Step 37 - cross_entropy_loss
def cross_entropy_loss(logits, targets):
    # TODO: Compute mean multi-class cross-entropy between logits and targets.
    m, c = logits.shape

    log_probs = torch.log_softmax(logits, dim=1)

    loss = torch.zeros(targets.shape,dtype=torch.float32)
    for i in range(m) : 
        class_i = targets[i]
        loss[i] = -log_probs[i,class_i]
    return torch.mean(loss,dim=0)

# Step 38 - mse_loss
def mse_loss(predictions, targets):
    # TODO: Compute mean squared error between predictions and targets
    predictions = torch.flatten(predictions)
    targets = torch.flatten(targets)

    loss = predictions - targets 
    loss = torch.pow(loss,2)
    return torch.mean(loss,dim=0)

# Step 39 - accuracy_metric
def accuracy_metric(logits, targets):
    # TODO: Return the fraction of argmax(logits) predictions matching targets.

    m, c  = logits.shape

    predictions = torch.argmax(logits,dim=1)
    
    return torch.mean((predictions==targets).float())

# Step 40 - mae_metric
def mae_metric(predictions, targets):
    # TODO: Compute mean absolute error between predicted and target continuous values.
    predictions = torch.flatten(predictions)
    targets = torch.flatten(targets)

    loss = predictions - targets 
    loss = torch.abs(loss)
    return torch.mean(loss,dim=0)

# Step 41 - gnn_train_step
def gnn_train_step(params, batch, forward_fn, loss_fn, lr):
    # TODO: Run one SGD training step and update params in-place...
    from torch.optim import SGD

    optimizer = SGD([p for p in params.values() if p.requires_grad],lr=lr)

    
    optimizer.zero_grad()

    out = forward_fn(params,batch)
    loss = loss_fn(out,batch['y'])

    loss.backward()
    optimizer.step()
    
    return {
        "loss" : loss.item(),
        "params" : params
    }

# Step 42 - train_node_classifier
def train_node_classifier(params, dataset, forward_fn, num_epochs, lr, mask_key='train_mask'):
    # TODO: Train a functional node-classification GNN for several epochs on a masked graph
    history = {
        "loss": [],
        "accuracy": []
    }
    mask = dataset[mask_key]
    x = dataset["x"]
    y = dataset["y"]
    edge_index = dataset["edge_index"]

    history = []

    batch = {
        "x": x,
        "edge_index": edge_index,
        "mask": mask,
        "y": y[mask]
    }

    def wrapped_forward(params, batch):
        logits = forward_fn(
            params,
            batch["x"],
            batch["edge_index"]
        )
        return logits[batch["mask"]]


    def masked_loss(logits, targets):
        return cross_entropy_loss(logits[mask],targets[mask])

    for epoch in range(num_epochs):
        step = gnn_train_step(
            params,
            batch,
            wrapped_forward,
            cross_entropy_loss,
            lr
        )

        params = step["params"]

        with torch.no_grad():
            logits = forward_fn(params, x, edge_index)

            acc = accuracy_metric(
                logits[mask],
                y[mask]
            )

        history.append({
            "loss": step["loss"],
            "accuracy": acc.item() if torch.is_tensor(acc) else float(acc)
        })

    return {
        "history": history,
        "params": params
    }

# Step 43 - train_graph_regressor (not yet solved)
# TODO: implement

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

