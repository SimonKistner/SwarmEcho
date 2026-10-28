"""Fixed-shape adaptive population and configurable one-hop vote transport.

In credits mode, YES emissions require a live peer or base link. Each emission
has a source sequence number and enters only on that communication tick. Relays retain
individual received packets and forward at most one per source per tick; the
sender does not retain packets for later reconnection. A relay can deliver an
earlier packet after the sender disconnects. Per-round receipt bits deduplicate
routes. NO does not retract delivered credits. Capacity rejections spend
batches without changing the round. In latest_decision mode, relays and base
retain only the newest timestamped YES/NO per source in their known round.
Every completed attempt (grant or capacity rejection) closes that round.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from swarmecho.core.config import AdaptSizeConfig


class VoteShapingState(NamedTuple):
    # n + 1 means no connected chain has been observed this episode.
    min_chain_agents: jax.Array
    decision_unnecessary: jax.Array  # Same receiver/source provenance as decision_step.
    withheld_voters: jax.Array  # This transition's eligible local NO choices.
    unnecessary_voters: jax.Array  # Responsible sources captured before base clearing.
    unnecessary_call: jax.Array  # A granted call; suppress its ordinary positive reward.


class VoteShapingContext(NamedTuple):
    budget: jax.Array  # Zero when neither an estimator nor an observed chain is available.
    base_target_known: jax.Array
    target_known: jax.Array
    chain_connected: jax.Array
    acting: jax.Array  # False for an already-terminal state.


def effective_reinforcement_budget(required_agents, min_chain_agents, n):
    """Use certified/observed sufficient populations, never a fabricated zero budget."""
    estimated = jnp.where(required_agents > 0, required_agents, n + 1)
    budget = jnp.minimum(estimated, min_chain_agents)
    return jnp.where(budget <= n, budget, 0).astype(jnp.int32)


class AdaptiveState(NamedTuple):
    vote: jax.Array
    ids: jax.Array
    next_id: jax.Array
    round: jax.Array
    known_round: jax.Array
    source_yes: jax.Array  # Next emission sequence; the source stores no packet bits.
    packet_round: jax.Array  # receivers (agents + base), source slots
    packet_count: jax.Array  # Credit tally, or a compatibility 0/1 latest-YES summary.
    packet_bits: jax.Array  # Packed per-sequence receipts; source rows stay empty.
    spent: jax.Array  # Credit-mode base receipts consumed by rejected calls this round.
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
    reference_callers: jax.Array  # [caller slot, recruit slot], scoped to lifetimes.
    reference_age: jax.Array  # Environment steps since the recruit's voted activation.
    vote_step: jax.Array  # Episode transition clock; independent of call rounds.
    decision_step: jax.Array  # [receiver (agents + base), source]; -1 means unknown.
    decision_yes: jax.Array  # Latest source decision, meaningful only with a timestamp.
    vote_shaping: VoteShapingState | None = None  # No extra array leaves when disabled.


def initial_state(n, initial_agents, zone_center, zone_valid=True, max_steps=1024,
                  vote_protocol="credits", vote_shaping_enabled=False):
    z = jnp.zeros(n, dtype=jnp.int32)
    b = jnp.zeros(n, dtype=jnp.bool_)
    packet_words = max(1, (max_steps + 31) // 32) if vote_protocol == "credits" else 1
    return AdaptiveState(
        b, jnp.where(jnp.arange(n) < initial_agents, jnp.arange(n), -1),
        jnp.int32(initial_agents), jnp.int32(0), z, z,
        jnp.full((n + 1, n), -1, dtype=jnp.int32),
        jnp.zeros((n + 1, n), dtype=jnp.int32),
        jnp.zeros((n + 1, n, packet_words), dtype=jnp.uint32),
        z, jnp.int32(0),
        zone_center, jnp.asarray(zone_valid), z, jnp.int32(0), jnp.int32(0),
        jnp.int32(0), b, b, b, jnp.bool_(False), jnp.int32(0),
        jnp.zeros((n, n), dtype=jnp.bool_), z,
        jnp.int32(0), jnp.full((n + 1, n), -1, dtype=jnp.int32),
        jnp.zeros((n + 1, n), dtype=jnp.bool_),
        (VoteShapingState(
            jnp.int32(n + 1), jnp.zeros((n + 1, n), dtype=jnp.bool_),
            b, b, jnp.bool_(False),
        ) if vote_shaping_enabled else None),
    )


def available_votes(a):
    received = jnp.where(a.packet_round[-1] == a.round, a.packet_count[-1], 0)
    return jnp.maximum(received - a.spent, 0)


def base_observation_feedback(a, active, connected_to_base, vote_holding, population_reference):
    """Instantaneous base summary for connected drones; no cached observation."""
    count = jnp.sum(active).astype(jnp.float32)
    qualified = active & (available_votes(a) >= vote_holding)
    summary = jnp.stack((
        jnp.sum(qualified).astype(jnp.float32) / jnp.maximum(count, 1.0),
        count / population_reference,
    ))
    return jnp.where((connected_to_base & active)[:, None], summary[None, :], 0.0)


def reinforcement_voters(previous, current):
    """All qualified sources captured for this transition's voted deployment.

    Grant references are captured before base receipts are cleared, regardless
    of whether reference rewards are enabled. Restrict to newly spawned recruits
    and a new call so old references and staggered initial spawns cannot pay.
    """
    return (jnp.any(current.reference_callers & current.spawned[None, :], axis=1)
            & (current.calls > previous.calls))


def apply_vote(vote, commands, mode):
    if mode == "hold":
        return commands >= 0.5
    # KEEP=0, YES=1, NO=2. The policy head samples these exact integers.
    return jnp.where(commands == 1, True, jnp.where(commands == 2, False, vote))


def _communicate_latest(values, alive, peer_links, base_links, current_round, stamp,
                        emission_unnecessary=None):
    """Merge newest source records from one pre-receive snapshot, never transitively.

    Fresh local choices enter neighboring receivers on this tick. A source does
    not queue its own past choices; its diagonal is only a temporary emission.
    The base receives records but sends only round announcements. Relays keep
    both YES and NO indefinitely within their round, including while isolated.
    """
    pr, pc, known, vote, timestamps, decisions = values[:6]
    context = values[6] if emission_unnecessary is not None else None
    n = alive.shape[0]
    peers = peer_links & alive[:, None] & alive[None, :]
    base = base_links & alive
    links = jnp.zeros((n + 1, n + 1), dtype=jnp.bool_)
    links = links.at[:n, :n].set(peers)
    links = links.at[:n, n].set(base).at[n, :n].set(base)
    links |= jnp.eye(n + 1, dtype=jnp.bool_)
    rounds = jnp.concatenate([known, current_round[None]])
    valid = pr == rounds[:, None]
    timestamps = jnp.where(valid, timestamps, -1)
    decisions &= valid & (timestamps >= 0)
    if context is not None:
        context &= valid & (timestamps >= 0)
    pr = jnp.broadcast_to(rounds[:, None], pr.shape)

    # A timestamp describes the source's choice, never the relay's arrival time.
    # Sampling both YES and NO each tick also refreshes an unchanged decision.
    indices = jnp.arange(n)
    can_send = alive & (jnp.any(peers, axis=1) | base)
    snapshot_time = timestamps.at[indices, indices].set(jnp.where(can_send, stamp, -1))
    snapshot_yes = decisions.at[indices, indices].set(vote & can_send)
    if context is not None:
        snapshot_context = context.at[indices, indices].set(emission_unnecessary & can_send)
    vote_links = links.at[:, n].set(False)
    vote_links = vote_links.at[jnp.arange(n + 1), jnp.arange(n + 1)].set(False)
    routes = vote_links & (rounds[:, None] == rounds[None, :])
    offered_time = jnp.where(routes[:, :, None], snapshot_time[None, :, :], -1)
    sender = jnp.argmax(offered_time, axis=1)
    newest = jnp.max(offered_time, axis=1)
    offered_yes = jnp.broadcast_to(snapshot_yes[None, :, :], offered_time.shape)
    newest_yes = jnp.take_along_axis(offered_yes, sender[:, None, :], axis=1)[:, 0, :]
    if context is not None:
        # Select provenance from the exact same source record as its timestamp
        # and YES/NO, so relays cannot reclassify a delayed approval.
        offered_context = jnp.broadcast_to(snapshot_context[None, :, :], offered_time.shape)
        newest_context = jnp.take_along_axis(offered_context, sender[:, None, :], axis=1)[:, 0, :]
    replace = newest > timestamps
    timestamps = jnp.where(replace, newest, timestamps)
    decisions = jnp.where(replace, newest_yes, decisions)
    if context is not None:
        context = jnp.where(replace, newest_context, context)
        context = context.at[indices, indices].set(False)
    timestamps = timestamps.at[indices, indices].set(-1)
    decisions = decisions.at[indices, indices].set(False)

    # The announcement also travels exactly one hop and overrides this tick's
    # action. Unknown records cannot qualify in the newly learned round.
    heard = jnp.max(jnp.where(links[:n], rounds[None, :], -1), axis=1)
    learned = alive & (heard > known)
    known = jnp.where(learned, heard, known)
    vote &= ~learned
    timestamps = timestamps.at[:n].set(jnp.where(learned[:, None], -1, timestamps[:n]))
    decisions = decisions.at[:n].set(jnp.where(learned[:, None], False, decisions[:n]))
    if context is not None:
        context = context.at[:n].set(context[:n] & ~learned[:, None])
    pr = pr.at[:n].set(jnp.where(learned[:, None], known[:, None], pr[:n]))
    # Compatibility summary for rewards, base feedback and older tally readers;
    # the protocol itself tracks a boolean decision, not accumulated credits.
    pc = ((timestamps >= 0) & decisions).astype(jnp.int32)
    result = (pr, pc, known, vote, timestamps, decisions)
    return result + (context,) if context is not None else result


def transition(a, active, pos, commands, peer_links, base_links, share_now,
               cfg: AdaptSizeConfig, *, shaping_context=None):
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
    decision_step = jnp.where(removed[None, :], -1, a.decision_step)
    decision_yes = a.decision_yes & ~removed[None, :]
    decision_step = decision_step.at[:n].set(jnp.where(alive[:, None], decision_step[:n], -1))
    decision_yes = decision_yes.at[:n].set(decision_yes[:n] & alive[:, None])
    vote_step = a.vote_step + 1
    shaping = a.vote_shaping
    vote_shaping_enabled = shaping_context is not None
    if vote_shaping_enabled:
        if shaping is None or shaping_context is None:
            raise ValueError("Enabled reinforcement vote shaping requires initialized state and decision context.")
        context_records = shaping.decision_unnecessary & ~removed[None, :]
        context_records = context_records.at[:n].set(context_records[:n] & alive[:, None])
        # Reserved initial arrivals already supply population without another call.
        occupied_for_vote = jnp.sum(alive) + jnp.sum((~active) & (a.ids >= 0))
        informed = (shaping_context.base_target_known & shaping_context.target_known
                    & shaping_context.acting)
        valid_budget = shaping_context.budget > 0
        # A simultaneous removal can break the pre-action chain. Do not use
        # that stale connectivity to punish a replacement within its budget.
        chain_available = shaping_context.chain_connected & ~jnp.any(removed)
        unnecessary = (shaping_context.base_target_known & shaping_context.acting
                       & (chain_available
                          | (valid_budget & (occupied_for_vote >= shaping_context.budget))))
        needs_recruit = (valid_budget & ~chain_available
                         & (occupied_for_vote < shaping_context.budget)
                         & (occupied_for_vote < n))
        emission_unnecessary = informed & unnecessary

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

    if cfg.vote_protocol == "latest_decision":
        values = (pr, pc, a.known_round, vote, decision_step, decision_yes)
        if vote_shaping_enabled:
            values += (context_records,)
        communicated = jax.lax.cond(
            share_now,
            lambda x: _communicate_latest(
                x, alive, peer_links, base_links, a.round, vote_step,
                emission_unnecessary if vote_shaping_enabled else None,
            ),
            lambda x: x, values,
        )
        pr, pc, known, vote, decision_step, decision_yes = communicated[:6]
        if vote_shaping_enabled:
            context_records = communicated[6]
    else:
        pr, pc, bits, source, known, vote = jax.lax.cond(
            share_now, communicate, lambda x: x, (pr, pc, bits, source, a.known_round, vote)
        )
    received = jnp.where(pr[n] == a.round, pc[n], 0)
    if cfg.vote_protocol == "latest_decision":
        ready = alive & (pr[n] == a.round) & (decision_step[n] >= 0) & decision_yes[n]
    else:
        ready = alive & ((received - spent) >= cfg.vote_holding)
    count = jnp.sum(alive)
    # Inactive initial drones retain reserved lifetime IDs until their normal
    # spawn-delay tick. A reinforcement cannot overwrite those reserved slots.
    reserved = (~active) & (a.ids >= 0)
    occupied = count + jnp.sum(reserved)
    approvals = jnp.sum(ready)
    # Full approval is attainable at 100%; lower fractions retain strict >.
    enough_votes = (approvals == count if cfg.quorum_fraction == 1.0
                    else approvals > cfg.quorum_fraction * count)
    quorum = (count > 0) & enough_votes
    held = jnp.where(quorum, a.quorum_steps + 1, 0)
    attempt = share_now & quorum & (held >= cfg.quorum_hold)
    grant = attempt & (occupied < n)
    reject = attempt & (occupied == n)
    rejected_voters = ready & reject
    if vote_shaping_enabled:
        # Reachability uses this tick's pre-action links, excluding removals.
        # A current YES is never charged for an older NO still at the base.
        round_alive = alive & (a.known_round == a.round)
        peers = peer_links & round_alive[:, None] & round_alive[None, :]
        reachable = base_links & round_alive
        reachable = jax.lax.fori_loop(
            0, n, lambda _, connected: connected | jnp.any(peers & connected[None, :], axis=1),
            reachable,
        )
        current_round = (a.known_round == a.round) & (known == a.known_round)
        withheld = (share_now & alive & informed & reachable & current_round
                    & needs_recruit & ~vote & ~attempt)
        shaping = shaping._replace(
            withheld_voters=withheld,
            unnecessary_voters=ready & context_records[n] & grant & unnecessary,
            unnecessary_call=grant & unnecessary,
        )
    # Each rejection consumes one fresh batch from every qualified voter.
    # Raw tallies and deployment round persist until a successful deployment.
    # Consume surplus too: an already-delivered backlog must never fund a
    # second penalty. These voters need vote_holding additional YES receipts.
    if cfg.vote_protocol == "credits":
        spent = jnp.where(rejected_voters, received, spent)
    slot = jnp.argmax((~alive) & ~reserved)
    spawned = (jnp.arange(n) == slot) & grant
    result_active = alive | spawned
    # Capture the qualified sources before the grant clears base receipts.
    # Remove both incoming and outgoing references on any lifetime boundary,
    # including decommission and replacement of the same slot on this step.
    continuing = alive & ~spawned
    reference_callers = a.reference_callers & continuing[:, None] & continuing[None, :]
    reference_callers |= ready[:, None] & spawned[None, :]
    reference_age = jnp.where(spawned, 0, a.reference_age + 1)
    # Latest decisions authorize one attempt. Rejecting also closes that round,
    # preventing repeated penalties from the same cached agreement.
    completed = grant | (reject & (cfg.vote_protocol == "latest_decision"))
    new_round = a.round + completed.astype(jnp.int32)
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
    pr = pr.at[n].set(jnp.where(completed, -1, pr[n]))
    pc = pc.at[n].set(jnp.where(completed, 0, pc[n]))
    bits = jnp.where(spawned[None, :, None], jnp.uint32(0), bits)
    bits = bits.at[:n].set(jnp.where(spawned[:, None, None], jnp.uint32(0), bits[:n]))
    bits = bits.at[n].set(jnp.where(completed, jnp.uint32(0), bits[n]))
    decision_step = jnp.where(spawned[None, :], -1, decision_step)
    decision_yes &= ~spawned[None, :]
    decision_step = decision_step.at[:n].set(jnp.where(spawned[:, None], -1, decision_step[:n]))
    decision_yes = decision_yes.at[:n].set(decision_yes[:n] & ~spawned[:, None])
    decision_step = decision_step.at[n].set(jnp.where(completed, -1, decision_step[n]))
    decision_yes = decision_yes.at[n].set(decision_yes[n] & ~completed)
    if vote_shaping_enabled:
        context_records &= ~spawned[None, :]
        context_records = context_records.at[:n].set(context_records[:n] & ~spawned[:, None])
        context_records = context_records.at[n].set(context_records[n] & ~completed)
        shaping = shaping._replace(decision_unnecessary=context_records)
    spent = jnp.where(completed, 0, spent)
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
        reference_callers=reference_callers, reference_age=reference_age,
        vote_step=vote_step, decision_step=decision_step, decision_yes=decision_yes,
        vote_shaping=shaping,
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
