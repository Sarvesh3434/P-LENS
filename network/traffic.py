"""[SIM] Traffic injection, congestion-aware routing, M/M/1-style delay.

Routing [ASSUMPTION]: shortest path with weight
    w_e = prop_latency_e * (1 + routing_alpha * u_e_prev) + hop_epsilon
where u_e_prev is the utilization from the previous routing pass (0 on the first
pass). `route_and_load` runs two passes: pass 1 places demand to measure per-link
utilization; pass 2 re-routes with those utilization weights. alpha = cfg.routing_alpha.

Delivery model [SIM][ASSUMPTION]: each flow's delivered share is its bottleneck
share over the path links:
    share_f = (min over e in path of min(1, mu / load_e)) * prod(rel_e)
with load_e the final placed load and mu = cfg.mm1_service_rate. An unloaded
path delivers everything (minus link reliability); an overloaded link delivers
mu / load of whatever traverses it. This keeps throughput in [0, 1] by
construction and makes throughput drop smoothly as load exceeds capacity.

Delay [SIM][ASSUMPTION]: queue term on a link
    q_e = (mu / (mu - lambda_e) - 1) / 20   if lambda_e < mu, else 1 (saturated)
with lambda_e = placed load. The division by 20 maps the idle→1 normalized
M/M/1 term into [0, 1]. q_e is stored on the edge and propagated into node
queue_delay (max over incident edges). The latency metric is the delivered-bytes
weighted mean of per-path sums of (prop_latency + q_e) of this pass.

Loss [SIM] = 1 - delivered/total: unreachable demand (endpoint down or no path)
plus per-flow bottleneck shortfall plus unreliability drops. Stored globally and
per node as each node's own undelivered in+out share.

`scale_demand` normalizes the demand matrix against *usable* bandwidth so
traffic_intensity means the same thing regardless of current failures.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from network.graph import EvolvingNetwork, canon


def _weighted_graph(net: EvolvingNetwork) -> nx.Graph:
    """Routing graph: alive nodes, non-failed edges, congestion-aware weights."""
    g = nx.Graph()
    alive = set(net.alive_nodes())
    g.add_nodes_from(alive)
    for (u, v), e in net.edges.items():
        if e.failed or u not in alive or v not in alive:
            continue
        w = e.prop_latency * (1.0 + net.cfg.routing_alpha * min(e.utilization, 1.0)) + net.cfg.hop_epsilon
        g.add_edge(u, v, weight=w, bandwidth=e.bandwidth)
    return g


def _all_sources_paths(g: nx.Graph) -> Dict[int, list]:
    """Shortest-path trees from every source, in two sparse calls.

    The first call uses congestion-aware latency weights for routing.  The
    second uses unit weights and returns the hop distance, which ``scale_demand``
    needs to convert a per-edge utilization target into a total offered load
    (a demand unit traverses several edges, so ignoring path length inflates
    every edge by the mean hop count).
    """
    nodes = list(g.nodes())
    if not nodes:
        return {}, {}, {}, {}, None
    idx = {v: k for k, v in enumerate(nodes)}
    rows, cols, data = [], [], []
    for a, b, d in g.edges(data=True):
        w = float(d.get("weight", 1.0))
        rows.extend((idx[a], idx[b]))
        cols.extend((idx[b], idx[a]))
        data.extend((w, w))
    m = len(nodes)
    mat = csr_matrix((data, (rows, cols)), shape=(m, m))
    dist, pred = sp_dijkstra(mat, directed=False, return_predecessors=True)
    hop_dist = sp_dijkstra(mat, directed=False, unweighted=True)
    return nodes, idx, dist, pred, hop_dist


def _path_from_pred(nodes, idx, pred, src: int, dst: int):
    if src == dst:
        return [src]
    s, k = idx[src], idx[dst]
    if s >= pred.shape[0] or k >= pred.shape[1] or pred[s, k] < 0:
        return None
    rev = [k]
    while k != s:
        k = int(pred[s, k])
        if k < 0:
            return None
        rev.append(k)
    return [nodes[x] for x in reversed(rev)]


def _mean_path_length(dist: np.ndarray) -> float:
    """Mean finite pairwise distance from the routing matrix (cheap APL)."""
    finite = np.isfinite(dist) & (dist > 0.0)
    if not finite.any():
        return float("nan")
    return float(dist[finite].mean())


def _stage_one(net: EvolvingNetwork) -> None:
    """First routing pass: place demand, keep only the utilization signal."""
    for e in net.edges.values():
        e.load = 0.0
        e.packet_load = 0.0
    g = _weighted_graph(net)
    demand = net.demand
    n = net.n
    nodes, idx, _dist, pred, _hop = _all_sources_paths(g)
    for i in range(n):
        if i not in idx:
            continue
        for j in range(n):
            vol = float(demand[i, j])
            if vol <= 0 or i == j:
                continue
            if j not in idx or i not in idx:
                continue
            path = _path_from_pred(nodes, idx, pred, i, j)
            if path is None:
                continue
            for a, b in zip(path, path[1:]):
                e = net.edges.get(canon(a, b))
                if e is not None:
                    e.load += vol
    # utilization from this first pass is the congestion signal for pass 2
    for e in net.edges.values():
        if not e.failed:
            e.utilization = min(4.0, e.load / max(e.bandwidth, 1.0))


def route_and_load(net: EvolvingNetwork, two_pass: bool = True) -> Dict[str, float]:
    """Place demand on congestion-aware shortest paths; update queues and metrics.

    ``two_pass=False`` skips the exploratory first pass and routes directly with
    the utilization state carried by ``net``.  It is used by counterfactual
    replays, where the factual and counterfactual branches are treated
    identically; one pass per branch keeps the estimator matched while roughly
    halving its cost.  Online control still uses the full two-pass router.
    """
    prev_util = {k: e.utilization for k, e in net.edges.items()}
    for e in net.edges.values():
        e.load = 0.0
        e.packet_load = 0.0
        e.utilization = 0.0
        e.queue_delay = 0.0
    for nd in net.nodes.values():
        nd.demand = 0.0
        nd.traffic_in = 0.0
        nd.traffic_out = 0.0
        nd.queue_delay = 0.0
        nd.loss = 0.0
        nd.utilization = 0.0
        nd.provisioned = 0.0
    # restore last pass's utilization so pass-1 weights are congestion-aware [SIM]
    for k, u in prev_util.items():
        if k in net.edges:
            net.edges[k].utilization = u

    if two_pass:
        _stage_one(net)  # [ASSUMPTION] two passes: pass 1 fills u_e_prev for pass 2

    g = _weighted_graph(net)  # weighted with (pass-1 or carried) utilization
    for e in net.edges.values():
        e.load = 0.0  # discard pass-1 loads; only their utilization signal is kept

    dropped = 0.0   # demand that could not be routed at all
    carried = 0.0   # demand placed on paths (before bottleneck/reliability shortfall)
    delivered_total = 0.0
    latency_acc = 0.0
    node_delivered = {i: 0.0 for i in net.nodes}
    flow_latencies: List[float] = []
    flow_losses: List[float] = []
    flows: List[Tuple[float, list, int, int]] = []  # (vol, edges_on_path, src, dst)
    demand = net.demand
    n = net.n

    # --- pass 2: final placement, collecting per-flow paths ---
    nodes, idx, dist, pred, hop_dist = _all_sources_paths(g)
    if hop_dist is not None:
        net.mean_hops = max(_mean_path_length(hop_dist), 1.0)
    for i in range(n):
        for j in range(n):
            vol = float(demand[i, j])
            if vol <= 0 or i == j:
                continue
            net.nodes[i].demand += vol
            net.nodes[i].traffic_out += vol
            net.nodes[j].traffic_in += vol
            if i not in idx or j not in idx:
                dropped += vol
                continue
            path = _path_from_pred(nodes, idx, pred, i, j)
            if path is None:
                dropped += vol
                continue
            edges_on_path = []
            ok = True
            for a, b in zip(path, path[1:]):
                e = net.edges.get(canon(a, b))
                if e is None or e.failed:
                    ok = False
                    break
                edges_on_path.append(e)
            if not ok:
                dropped += vol
                continue
            for e in edges_on_path:
                e.load += vol
            carried += vol
            flows.append((vol, edges_on_path, i, j))

    # --- queues from final placed loads [ASSUMPTION: M/M/1 normalized] ---
    mu = net.cfg.mm1_service_rate
    for e in net.edges.values():
        if e.failed:
            continue
        e.utilization = min(4.0, e.load / max(e.bandwidth, 1.0))
        if e.load >= mu:
            q = 1.0  # saturated
        else:
            q = (mu / max(mu - e.load, 1e-9) - 1.0) / 20.0  # normalized M/M/1 [ASSUMPTION]
            q = min(q, 1.0)
        e.queue_delay = q
        e.packet_load = e.load
        net.nodes[e.u].queue_delay = max(net.nodes[e.u].queue_delay, q)
        net.nodes[e.v].queue_delay = max(net.nodes[e.v].queue_delay, q)

    # --- per-flow bottleneck delivery + latency [SIM][ASSUMPTION] ---
    for vol, edges_on_path, i, j in flows:
        bottleneck = 1.0
        rel_prod = 1.0
        path_lat = 0.0
        for e in edges_on_path:
            bottleneck = min(bottleneck, mu / max(e.load, 1e-9))
            rel_prod *= min(max(e.reliability, 0.0), 1.0)
            path_lat += e.prop_latency + e.queue_delay
        share = min(1.0, bottleneck) * rel_prod
        delivered = vol * share
        delivered_total += delivered
        node_delivered[i] += delivered
        node_delivered[j] += delivered
        latency_acc += path_lat * delivered
        flow_latencies.append(path_lat)
        flow_losses.append(1.0 - share)

    for i, nd in net.nodes.items():
        bw = 0.0
        max_u = 0.0
        for nb in net.G.neighbors(i) if i in net.G else []:
            e = net.edges.get(canon(i, nb))
            if e and not e.failed:
                bw += e.bandwidth
                max_u = max(max_u, min(e.utilization, 1.0))
        nd.provisioned = bw
        nd.utilization = max_u if not nd.failed else 1.0

    total_demand = float(demand.sum())
    # [SIM] throughput = delivered share of demand; loss = undelivered share
    throughput = min(1.0, max(0.0, delivered_total / max(total_demand, 1e-9)))
    loss_frac = 1.0 - throughput
    util_sum = 0.0
    edge_count = 0
    for e in net.edges.values():
        if e.failed:
            continue
        util_sum += min(e.utilization, 1.0)
        edge_count += 1
    for i, nd in net.nodes.items():
        own = float(nd.demand + net.nodes[i].traffic_in)
        # per-node loss = own undelivered in+out share [SIM]
        denom = nd.traffic_out + net.nodes[i].traffic_in
        nd.loss = min(1.0, max(0.0, 1.0 - node_delivered[i] / max(denom, 1e-9))) if not nd.failed else 1.0

    avg_lat = latency_acc / max(delivered_total, 1e-9)
    if flow_latencies:
        lat_arr = np.asarray(flow_latencies, dtype=float)
        loss_arr = np.asarray(flow_losses, dtype=float)
        lat_p95 = float(np.percentile(lat_arr, 95))
        lat_p99 = float(np.percentile(lat_arr, 99))
        sla_lat = net.cfg.sla_latency_ms / 1000.0
        sla_lat_viol = float(np.mean(lat_arr > sla_lat))
        sla_loss_viol = float(np.mean(loss_arr > net.cfg.sla_loss))
        sla_viol = float(np.mean((lat_arr > sla_lat) | (loss_arr > net.cfg.sla_loss)))
    else:
        lat_p95 = lat_p99 = 0.0
        sla_lat_viol = sla_loss_viol = sla_viol = 0.0
    metrics = {
        "latency_p95": lat_p95,
        "latency_p99": lat_p99,
        "sla_latency_violation": sla_lat_viol,
        "sla_loss_violation": sla_loss_viol,
        "sla_violation": sla_viol,
        "latency": float(avg_lat),
        "throughput": float(throughput),
        "loss": float(loss_frac),
        "congestion": float(util_sum / max(edge_count, 1)),
        "avg_util": float(util_sum / max(edge_count, 1)),
        "carried": float(carried),
        "delivered": float(delivered_total),
        "demand": float(total_demand),
        "n_edges": float(edge_count),
        "connected": float(1.0 if (g.number_of_nodes() <= 1 or nx.is_connected(g)) else 0.0),
        "total_demand": float(total_demand),
        "total_loss": float(max(0.0, total_demand - delivered_total)),
    }
    if g.number_of_nodes() > 1 and nx.is_connected(g):
        metrics["avg_path_length"] = _mean_path_length(dist)
    else:
        metrics["avg_path_length"] = float("nan")
    net.last_metrics = metrics
    return metrics


def scale_demand(matrix: np.ndarray, net: EvolvingNetwork, target_util: float) -> np.ndarray:
    """Scale so mean per-edge load ≈ target_util × usable bandwidth.

    A demand unit is carried over several edges, so the offered load must be
    divided by the mean hop count; otherwise a 55% per-edge target becomes a
    165% load on a 3-hop topology and every policy is saturated.

    Both the bandwidth budget and the hop count are measured on the *reference*
    topology (minus currently failed elements), never on the controller's
    mutated topology.  If they were measured on the current graph, a policy
    that adds links would be rewarded with more absolute demand, which cancels
    the capacity it just bought and makes policy comparisons unfair.
    """
    busy_nodes = {i for i, nd in net.nodes.items() if nd.failed}
    usable_bw = 0.0
    for u, v in net.reference.edges():
        if u in busy_nodes or v in busy_nodes:
            continue
        e = net.edges.get(canon(u, v))
        if e is not None and e.failed:
            continue
        usable_bw += float(e.bandwidth) if e is not None else float(net.cfg.default_bandwidth)
    total_bw = usable_bw or 1.0
    s = float(matrix.sum())
    if s <= 0:
        return matrix
    hops = max(float(getattr(net, "ref_mean_hops", 3.0) or 3.0), 1.0)
    factor = (target_util * total_bw) / (s * hops)
    return matrix * factor
