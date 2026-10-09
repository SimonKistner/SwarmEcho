"""Fixed-shape adaptive population and synchronous one-hop vote transport.

YES emissions require a live peer or base link. Each emission has a source
sequence number and is injected only on that communication tick. Relays retain
individual received packets and forward at most one per source per tick; the
sender does not retain packets for later reconnection. A relay can deliver an
earlier packet after the sender disconnects. Per-round receipt bits deduplicate
routes. NO does not retract delivered credits. Capacity rejections spend
batches without changing the round.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from swarmecho.core.config import AdaptSizeConfig


class AdaptiveState(NamedTuple):
    vote: jax.Array
    ids: jax.Array
    next_id: jax.Array
    round: jax.Array
    known_round: jax.Array
    source_yes: jax.Array  # Next emission sequence; the source stores no packet bits.
    packet_round: jax.Array  # receivers (agents + base), source slots
    packet_count: jax.Array
    packet_bits: jax.Array  # Packed per-sequence receipts; source rows stay empty.
    spent: jax.Array  # Base receipts consumed by rejected calls this round.
    quorum_steps: jax.Array
    zone_center: jax.Array
    zone_valid: jax.Array
    dwell: jax.Array
    calls: jax.Array
    decommissions: jax.Array
    rejected_calls: jax.Array
    spawned: jax.Array
    decommissioned: jax.Array
    rejected_voters: jax.Array
    all_decommissioned: jax.Array
    peer_credit_count: jax.Array


def initial_state(n, initial_agents, zone_center, zone_valid=True, max_steps=1024):
    z = jnp.zeros(n, dtype=jnp.int32)
    b = jnp.zeros(n, dtype=jnp.bool_)
    packet_words = max(1, (max_steps + 31) // 32)
    return AdaptiveState(
        b, jnp.where(jnp.arange(n) < initial_agents, jnp.arange(n), -1),
        jnp.int32(initial_agents), jnp.int32(0), z, z,
        jnp.full((n + 1, n), -1, dtype=jnp.int32),
        jnp.zeros((n + 1, n), dtype=jnp.int32),
        jnp.zeros((n + 1, n, packet_words), dtype=jnp.uint32),
        z, jnp.int32(0),
        zone_center, jnp.asarray(zone_valid), z, jnp.int32(0), jnp.int32(0),
        jnp.int32(0), b, b, b, jnp.bool_(False), jnp.int32(0),
    )


def available_votes(a):
    received = jnp.where(a.packet_round[-1] == a.round, a.packet_count[-1], 0)
    return jnp.maximum(received - a.spent, 0)


def apply_vote(vote, commands, mode):
    if mode == "hold":
        return commands >= 0.5
    # KEEP=0, YES=1, NO=2. The policy head samples these exact integers.
    return jnp.where(commands == 1, True, jnp.where(commands == 2, False, vote))


def transition(a, active, pos, commands, peer_links, base_links, share_now,
               cfg: AdaptSizeConfig):
    """Return next protocol state and active mask; every array retains its shape."""
    n = active.shape[0]
    inside = (jnp.linalg.norm(pos - a.zone_center, axis=-1)
              <= cfg.decommission_radius_m) & active & a.zone_valid
    dwell = jnp.where(inside, a.dwell + 1, 0)
    removed = active & (dwell >= cfg.decommission_hold_steps)
    alive = active & ~removed
    vote = apply_vote(a.vote, commands, cfg.vote_mode) & alive
    # Physical removal invalidates that lifetime's queued traffic everywhere.
    # Slot reuse therefore cannot make an old identity's receipts count again.
    pr = jnp.where(removed[None, :], -1, a.packet_round)
    pc = jnp.where(removed[None, :], 0, a.packet_count)
    bits = jnp.where(removed[None, :, None], jnp.uint32(0), a.packet_bits)
    pr = pr.at[:n].set(jnp.where(alive[:, None], pr[:n], -1))
    pc = pc.at[:n].set(jnp.where(alive[:, None], pc[:n], 0))
    bits = bits.at[:n].set(jnp.where(alive[:, None, None], bits[:n], jnp.uint32(0)))
    spent = jnp.where(alive, a.spent, 0)
    source = jnp.where(alive, a.source_yes, 0)

    def communicate(values):
        pr, pc, bits, source, known, vote = values
        peers = peer_links & alive[:, None] & alive[None, :]
        base = base_links & alive
        links = jnp.zeros((n + 1, n + 1), dtype=jnp.bool_)
        links = links.at[:n, :n].set(peers)
        links = links.at[:n, n].set(base).at[n, :n].set(base)
        links |= jnp.eye(n + 1, dtype=jnp.bool_)
        # Only this tick's emission enters the network. The source row never
        # stores packets, so a later reconnection cannot replay earlier YESes.
        can_send = jnp.any(peers, axis=1) | base
        words = bits.shape[-1]
        emit = vote & can_send & (source < words * 32)
        word_index = source // 32
        bit_value = jnp.left_shift(jnp.uint32(1), (source % 32).astype(jnp.uint32))
        fresh_bits = jnp.where(
            emit[:, None] & (jnp.arange(words)[None, :] == word_index[:, None]),
            bit_value[:, None], jnp.uint32(0),
        )
        source = source + emit.astype(jnp.int32)

        receiver_round = jnp.concatenate([known, a.round[None]])
        current_receipts = pr == receiver_round[:, None]
        bits = jnp.where(current_receipts[:, :, None], bits, jnp.uint32(0))
        pc = jnp.where(current_receipts, pc, 0)
        pr = jnp.broadcast_to(receiver_round[:, None], (n + 1, n))

        # Vote packets use the pre-receive snapshot: one physical hop per tick.
        # The base never sends vote packets back out; it only acknowledges rounds.
        vote_links = links.at[:, n].set(False)
        vote_links = vote_links.at[jnp.arange(n + 1), jnp.arange(n + 1)].set(False)
        same_round = receiver_round[:, None] == receiver_round[None, :]
        offered = jnp.where(
            (vote_links & same_round)[:, :, None, None],
            bits[None, :, :, :], jnp.uint32(0),
        )
        forwarded = jax.lax.reduce(
            offered, jnp.uint32(0), jax.lax.bitwise_or, dimensions=(1,),
        )
        fresh_links = vote_links[:, :n] & (receiver_round[:, None] == known[None, :])
        fresh = jnp.where(fresh_links[:, :, None], fresh_bits[None, :, :], jnp.uint32(0))
        # Prioritize a newly emitted vote so it actually enters the network on
        # its emission tick; relayed backlog remains at its previous relay.
        candidate = jnp.where(jnp.any(fresh != 0, axis=-1)[:, :, None], fresh, forwarded)
        missing = candidate & ~bits
        next_word = jnp.argmax(missing != 0, axis=-1)
        selected = jnp.take_along_axis(missing, next_word[:, :, None], axis=-1)[:, :, 0]
        lowest_bit = selected & (~selected + jnp.uint32(1))
        one_packet = jnp.where(
            jnp.arange(words)[None, None, :] == next_word[:, :, None],
            lowest_bit[:, :, None], jnp.uint32(0),
        )
        # Receiving one's own packet would turn the source into a store.
        one_packet = jnp.where(
            jnp.arange(n + 1)[:, None, None] == jnp.arange(n)[None, :, None],
            jnp.uint32(0), one_packet,
        )
        bits |= one_packet
        pc += jnp.any(one_packet != 0, axis=-1).astype(jnp.int32)
        # Fulfillment announcements use the same pre-receive snapshot and hops.
        announcements = jnp.concatenate([known, a.round[None]])
        heard = jnp.max(jnp.where(links[:n], announcements[None, :], -1), axis=1)
        learned = alive & (heard > known)
        known = jnp.where(learned, heard, known)
        vote = vote & ~learned
        source = jnp.where(learned, 0, source)
        bits = bits.at[:n].set(jnp.where(learned[:, None, None], jnp.uint32(0), bits[:n]))
        pc = pc.at[:n].set(jnp.where(learned[:, None], 0, pc[:n]))
        pr = pr.at[:n].set(jnp.where(learned[:, None], known[:, None], pr[:n]))
        # Acknowledgment wins over a simultaneous policy YES on this transition.
        return pr, pc, bits, source, known, vote

    pr, pc, bits, source, known, vote = jax.lax.cond(
        share_now, communicate, lambda x: x, (pr, pc, bits, source, a.known_round, vote)
    )
    received = jnp.where(pr[n] == a.round, pc[n], 0)
    ready = alive & ((received - spent) >= cfg.vote_holding)
    count = jnp.sum(alive)
    # Inactive initial drones retain reserved lifetime IDs until their normal
    # spawn-delay tick. A reinforcement cannot overwrite those reserved slots.
    reserved = (~active) & (a.ids >= 0)
    occupied = count + jnp.sum(reserved)
    quorum = (count > 0) & (jnp.sum(ready) > cfg.quorum_fraction * count)
    held = jnp.where(quorum, a.quorum_steps + 1, 0)
    attempt = share_now & quorum & (held >= cfg.quorum_hold)
    grant = attempt & (occupied < n)
    reject = attempt & (occupied == n)
    rejected_voters = ready & reject
    # Each rejection consumes one fresh batch from every qualified voter.
    # Raw tallies and deployment round persist until a successful deployment.
    # Consume surplus too: an already-delivered backlog must never fund a
    # second penalty. These voters need vote_holding additional YES receipts.
    spent = jnp.where(rejected_voters, received, spent)
    slot = jnp.argmax((~alive) & ~reserved)
    spawned = (jnp.arange(n) == slot) & grant
    result_active = alive | spawned
    new_round = a.round + grant.astype(jnp.int32)
    ids = jnp.where(removed, -1, a.ids)
    ids = jnp.where(spawned, a.next_id, ids)
    # Newborn learns protocol round at the station, but no task knowledge/memory.
    known = jnp.where(spawned, new_round, known)
    vote &= ~spawned
    source = jnp.where(spawned, 0, source)
    pr = jnp.where(spawned[None, :], -1, pr)
    pc = jnp.where(spawned[None, :], 0, pc)
    pr = pr.at[:n].set(jnp.where(spawned[:, None], -1, pr[:n]))
    pc = pc.at[:n].set(jnp.where(spawned[:, None], 0, pc[:n]))
    pr = pr.at[n].set(jnp.where(grant, -1, pr[n]))
    pc = pc.at[n].set(jnp.where(grant, 0, pc[n]))
    bits = jnp.where(spawned[None, :, None], jnp.uint32(0), bits)
    bits = bits.at[:n].set(jnp.where(spawned[:, None, None], jnp.uint32(0), bits[:n]))
    bits = bits.at[n].set(jnp.where(grant, jnp.uint32(0), bits[n]))
    spent = jnp.where(grant, 0, spent)
    return a._replace(
        vote=vote, ids=ids, next_id=a.next_id + grant.astype(jnp.int32),
        round=new_round, known_round=known, source_yes=source,
        packet_round=pr, packet_count=pc, packet_bits=bits, spent=spent,
        quorum_steps=jnp.where(attempt, 0, held),
        dwell=jnp.where(result_active & ~spawned, dwell, 0),
        calls=a.calls + grant.astype(jnp.int32),
        decommissions=a.decommissions + jnp.sum(removed, dtype=jnp.int32),
        rejected_calls=a.rejected_calls + reject.astype(jnp.int32),
        spawned=spawned, decommissioned=removed, rejected_voters=rejected_voters,
        all_decommissioned=~jnp.any(result_active),
    ), result_active


def lifetime_ended(previous, current):
    """Per-slot trajectory boundary, including immediate decommission/replacement."""
    if previous.adaptive is None:
        return previous.active & ~current.active
    return previous.active & (
        ~current.active | (previous.adaptive.ids != current.adaptive.ids)
    )


def policy_resets(state):
    return ~state.active if state.adaptive is None else (~state.active | state.adaptive.spawned)
