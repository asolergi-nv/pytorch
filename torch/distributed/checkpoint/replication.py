# mypy: allow-untyped-defs
"""Replica-aware checkpoint loading.

EXPERIMENTAL. When a model is sharded across many ranks, some tensors are
replicated: every data-parallel rank holds the same tensor-parallel shard. A
checkpoint stores each such shard once, so a plain ``dcp.load`` has every
replica holder read the same bytes, and storage sees ``replication_factor``
times more read traffic than the checkpoint is worth.

:class:`ReplicaAwareStorageReader` decorates any :class:`StorageReader` so that
each distinct piece of checkpoint data is read **once per replication group**
and distributed to the other members over the interconnect. Megatron-LM users
know this feature as ``FullyParallelLoadStrategyWrapper``.

    from torch.distributed.checkpoint.replication import ReplicaAwareStorageReader

    dcp.load(
        state_dict,
        storage_reader=ReplicaAwareStorageReader(
            FileSystemReader(path), replication_group=dp_group
        ),
    )

Nothing about the on-disk format changes, and the loaded state dict is
identical to what the wrapped reader would have produced on its own.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import os
import time
import types
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.metadata import (
    Metadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
)
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter, WriteResult
from torch.futures import Future


__all__ = [
    "ReplicaAwareStorageReader",
    "ReplicaAwareStorageWriter",
    "ReplicationOptions",
]

logger: logging.Logger = logging.getLogger(__name__)

# Which stored bytes a read request wants. Deliberately excludes the
# destination: two ranks with the same key receive byte-identical tensors even
# if they copy them into different local offsets, which is what makes the
# deduplication correct under resharding.
_ReadKey = tuple[MetadataIndex, torch.Size, torch.Size]


def _read_key(item: ReadItem) -> _ReadKey:
    return (item.storage_index, item.storage_offsets, item.lengths)


def _sort_key(key: _ReadKey) -> tuple:
    return (key[0].fqn, tuple(key[0].offset or ()), tuple(key[1]), tuple(key[2]))


def _read_nbytes(key: _ReadKey, metadata: Metadata) -> int:
    md = metadata.state_dict_metadata[key[0].fqn]
    if not isinstance(md, TensorStorageMetadata):
        return 1
    numel = 1
    for size in key[2]:
        numel *= size
    return numel * torch._utils._element_size(md.properties.dtype)


# Arena offsets are aligned to this so a uint8 slice can be reinterpreted as any
# dtype: Tensor.view(dtype) requires a storage offset divisible by the itemsize.
_ALIGN = 16


def _tensor_md(key: _ReadKey, metadata: Metadata) -> TensorStorageMetadata:
    md = metadata.state_dict_metadata[key[0].fqn]
    if not isinstance(md, TensorStorageMetadata):
        raise AssertionError(f"expected a tensor entry for {key[0].fqn}")
    return md


def _bucket(entries, metadata, limit):
    """Group an exchange schedule into runs sharing a source rank and dtype.

    Regrouping, not just splitting: the schedule is ordered by key, and the
    election interleaves owners to balance bytes, so consecutive entries rarely
    share a source. Walking it in order would put nearly every key in a bucket
    of its own. Collecting by source first is what makes coalescing possible.

    Every rank derives the groups from the same entries and the same metadata,
    so every rank yields the same runs in the same order, which is what lets
    each run be broadcast collectively.
    """
    groups: dict[tuple[int, torch.dtype], list[_ReadKey]] = defaultdict(list)
    for src, key in entries:
        groups[(src, _tensor_md(key, metadata).properties.dtype)].append(key)

    for src, dtype in sorted(groups, key=lambda g: (g[0], str(g[1]))):
        run: list[_ReadKey] = []
        total = 0
        for key in groups[(src, dtype)]:
            nbytes = _read_nbytes(key, metadata)
            if run and total + nbytes > limit:
                yield src, dtype, run
                run, total = [], 0
            run.append(key)
            total += nbytes
        if run:
            yield src, dtype, run


def _waves(entries, metadata, wave_bytes):
    """Split the schedule into waves that fit one arena slot per source.

    Yields ``{source rank: [(key, byte offset, nbytes), ...]}``. The layout
    follows only from the schedule and the metadata, so a reader computes where
    a peer put a chunk without being told: the arena needs no address table,
    only one base pointer per rank.

    A chunk larger than a slot still gets a wave to itself rather than being
    dropped; the arena has to be at least as big as the largest chunk.
    """
    by_src: dict[int, list[_ReadKey]] = defaultdict(list)
    for src, key in entries:
        by_src[src].append(key)
    sources = sorted(by_src)
    cursor = dict.fromkeys(sources, 0)

    while any(cursor[s] < len(by_src[s]) for s in sources):
        wave: dict[int, list[tuple[_ReadKey, int, int]]] = {}
        for src in sources:
            keys, index, packed, offset = by_src[src], cursor[src], [], 0
            while index < len(keys):
                nbytes = _read_nbytes(keys[index], metadata)
                if packed and offset + nbytes > wave_bytes:
                    break
                packed.append((keys[index], offset, nbytes))
                offset = -(-(offset + nbytes) // _ALIGN) * _ALIGN
                index += 1
            cursor[src] = index
            if packed:
                wave[src] = packed
        yield wave


def _fingerprint(payload: str) -> int:
    """63-bit digest of a string, stable across ranks and processes.

    `hash()` is salted per process, so it cannot be compared across ranks. Wider
    than a CRC deliberately: a digest match is now the basis for *skipping* a
    gather and assuming every rank wants the same keys, so a collision would be
    a silent mis-election rather than a stale cache. 63 bits keeps it inside a
    signed int64, which is what the collective moves.
    """
    return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big") >> 1


def _digest(keys: list[_ReadKey]) -> int:
    """Stable fingerprint of a key set."""
    return _fingerprint("|".join(repr(_sort_key(k)) for k in sorted(keys, key=_sort_key)))


def _collective_device(group) -> torch.device:
    """Where to put the handful of integers a digest exchange moves.

    CPU whenever the backend supports it. There is no bandwidth argument for the
    accelerator at this size, and it avoids a real hazard: the save election
    runs on the async checkpoint thread, which does not inherit the per-rank
    CUDA device, so `torch.cuda.current_device()` there can name device 0 on
    every rank and the collective deadlocks.
    """
    supported = getattr(group or dist.group.WORLD, "_device_types", [])
    if torch.device("cpu") in supported or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def _gather_digests(value: int, group, group_size: int) -> tuple[int, ...]:
    """all_gather one integer per rank without pickling anything.

    `all_gather_object` would serialise a Python object, and on a real plan that
    pickling *is* the cost: 1.8 s of a 14 s 16B load for a two-rank group moving
    about 480 KB. A fixed-shape int64 tensor is something the collective moves
    directly. Written as a masked all_reduce because every backend supports it.
    """
    slot = torch.zeros(group_size, dtype=torch.int64, device=_collective_device(group))
    slot[dist.get_rank(group)] = value
    dist.all_reduce(slot, group=group)
    return tuple(int(v) for v in slot)


def _payload_digest(payload: list[tuple[MetadataIndex, int]]) -> int:
    """Stable fingerprint of a write plan's (index, size) pairs."""
    return _fingerprint("|".join(sorted(f"{_write_sort_key(i)}:{n}" for i, n in payload)))


def _elect_owners(
    needers: dict[Any, list[int]],
    sizes: dict[Any, int],
    num_ranks: int,
    tiebreak: Callable[[Any], tuple],
) -> dict[Any, int]:
    """Assign one owner per key, balancing bytes. Deterministic given the input.

    Rarest first (a key available on only one rank has no alternative), then
    largest first, then a stable tie break, so every rank computing this from
    the same input reaches the same answer without further agreement. Shared by
    the read and write paths, which differ only in what a key is.
    """
    assigned = [0] * num_ranks
    owners: dict[Any, int] = {}
    order = sorted(
        needers, key=lambda k: (len(needers[k]), -sizes[k], tiebreak(k))
    )
    for key in order:
        chosen = min(needers[key], key=lambda r: (assigned[r], r))
        owners[key] = chosen
        assigned[chosen] += sizes[key]
    return owners


def _write_key(item: WriteItem) -> MetadataIndex:
    """DCP's own notion of "the same data": what `dedup_save_plans` keys on."""
    return item.index


def _write_sort_key(index: MetadataIndex) -> tuple:
    return (index.fqn, tuple(index.offset or ()))


def _write_nbytes(item: WriteItem) -> int:
    """Bytes this item writes.

    `WriteItem.tensor_storage_size()`, which DCP's own `dedup_save_plans` uses,
    multiplies out the *global* shape rather than the local chunk, so every
    chunk of an N-way sharded tensor is weighted N times too heavily.
    """
    if item.tensor_data is None or item.tensor_data.chunk is None:
        return 1
    numel = 1
    for size in item.tensor_data.chunk.sizes:
        numel *= size
    return numel * torch._utils._element_size(item.tensor_data.properties.dtype)


# Subgroups rebuilt from a rank partition, keyed by the partition. Module level
# so a long-lived process builds each group once.
_GROUP_CACHE: dict[tuple[tuple[int, ...], ...], Any] = {}


def _resolve_group(partition: tuple[tuple[int, ...], ...]):
    """Rebuild this rank's subgroup from a picklable rank partition.

    ``dist.new_group`` is a collective over the whole world, so every rank must
    call it for every entry of the partition in the same order.
    """
    if partition in _GROUP_CACHE:
        return _GROUP_CACHE[partition]
    me = dist.get_rank()
    mine = None
    for ranks in partition:
        group = dist.new_group(ranks=list(ranks))
        if me in ranks:
            mine = group
    _GROUP_CACHE[partition] = mine
    return mine


@dataclass
class ReplicationOptions:
    """Knobs for :class:`ReplicaAwareStorageReader`.

    Args:
        election: where one reader per replica set is chosen.
            ``"coordinator"`` (the default) does it on rank 0 inside the plan
            reduce-scatter DCP already runs, so it adds no collective at all.
            ``"group"`` does it inside the replication group with one
            fixed-size ``all_gather`` of digests -- key lists are only shipped
            when the group turns out not to be uniformly replicated. It
            additionally shrinks the world-wide plan gather by roughly the group
            size, and is the only mode that works with ``no_dist=True``. There is deliberately no adaptive setting:
            the two modes must agree across ranks, and whether a real
            coordinator exists cannot be determined locally on rank 0.
        exchange_device: device the broadcast buffers live on. Defaults to the
            current accelerator when the replication group supports it, else CPU.
        dedup_bytes: deduplicate non-tensor (``BYTE_IO``) items too. Not
            implemented yet; these are small and every rank keeps reading its own.
        validate: check that every read request this rank made was satisfied.
            Costs a dict lookup per item; useful while bringing up a new layout.
        enable_plan_caching: reuse the previous election when no rank's read
            requests changed. Mirrors ``DefaultSavePlanner(enable_plan_caching=)``:
            the first call pays full price, later ones exchange only a checksum.
        bucket_bytes: coalesce the exchange into broadcasts of at most this many
            bytes. ``0`` (the default) keeps one broadcast per chunk. A large
            checkpoint has thousands of chunks, and at that count the per-
            collective cost dominates the bytes; bucketing trades it for one
            flat buffer per bucket. Loaded values are unaffected either way.
            Only applies to ``transport="broadcast"``.
        transport: how a chunk gets from its reader to the other group members.
            ``"broadcast"`` (the default) uses ``dist.broadcast``, so every
            group member joins every exchange whether or not it wants the data.
            ``"nixl"`` uses one-sided RDMA reads through NVIDIA NIXL, so only
            ranks that want a chunk move bytes for it, and the staging buffers
            are one bounded arena instead of one allocation per chunk. Requires
            the ``nixl`` package.
        arena_bytes: total size of the registered staging arena for
            ``transport="nixl"``, split into one slot per group member. Bounds
            the transport's memory: bigger means fewer waves and fewer barriers.
    """

    election: str = "coordinator"
    exchange_device: Optional[torch.device] = None
    dedup_bytes: bool = False
    validate: bool = False
    enable_plan_caching: bool = False
    bucket_bytes: int = 0
    transport: str = "broadcast"
    arena_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        if self.election not in ("coordinator", "group"):
            raise ValueError(
                f"election must be 'coordinator' or 'group', got {self.election!r}"
            )
        if self.dedup_bytes:
            raise NotImplementedError("dedup_bytes is not implemented yet")
        if self.bucket_bytes < 0:
            raise ValueError(f"bucket_bytes must be >= 0, got {self.bucket_bytes}")
        if self.transport not in ("broadcast", "nixl"):
            raise ValueError(
                f"transport must be 'broadcast' or 'nixl', got {self.transport!r}"
            )
        if self.arena_bytes < _ALIGN:
            raise ValueError(f"arena_bytes must be >= {_ALIGN}, got {self.arena_bytes}")


@dataclass
class _Tag:
    """Rides DCP's existing plan gather so the coordinator learns the groups."""

    inner: Any
    group_ranks: tuple[int, ...]


@dataclass
class _Schedule:
    """Ordered exchange schedule, identical on every rank of a group.

    Carries only ``(source global rank, read key)`` per entry, so it stays small
    on the wire; each rank re-derives its own role by matching its local items.
    Keys wanted by a single rank are omitted: that rank reads them and there is
    nobody to send them to.
    """

    inner: Any = None
    entries: list[tuple[int, _ReadKey]] = field(default_factory=list)


class _CapturingPlanner:
    """Planner proxy that snapshots the tensors this rank has to broadcast.

    We cannot re-derive them from ``resolve_tensor`` afterwards: that method
    returns a *destination* for the storage layer to fill, and a planner is
    free to hand back a scratch buffer and only move the data into the real
    tensor in ``commit_tensor``. Re-resolving would broadcast an empty buffer.
    """

    def __init__(
        self, planner: LoadPlanner, wanted: set[_ReadKey], device: torch.device
    ) -> None:
        self._inner = planner
        self._wanted = wanted
        self._device = device
        self.captured: dict[_ReadKey, torch.Tensor] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        return self._inner.resolve_tensor(read_item)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        key = _read_key(read_item)
        if key in self._wanted:
            local = tensor.detach()
            # `.to` already copies across devices; only clone when it would not
            local = (
                local.to(self._device)
                if local.device != self._device
                else local.clone()
            )
            self.captured[key] = local.contiguous()
        self._inner.commit_tensor(read_item, tensor)

    def load_bytes(self, read_item: ReadItem, value) -> None:
        self._inner.load_bytes(read_item, value)

    def resolve_bytes(self, read_item: ReadItem):
        return self._inner.resolve_bytes(read_item)


class ReplicaAwareStorageReader(StorageReader):
    """Read each replicated chunk once per group, then broadcast it.

    EXPERIMENTAL.

    Args:
        storage_reader: the reader to decorate. Any implementation works; this
            class only reorganises *who* reads what.
        replication_group: the process group whose members hold replicas of each
            other, typically the data-parallel group. Defaults to the world,
            which is only useful if the state dict is fully replicated.
        replication_ranks: alternative to ``replication_group`` for callers that
            cannot hold a live ``ProcessGroup`` (it is not picklable). Give the
            full partition of the world, e.g. ``[[0, 2], [1, 3]]``; the group is
            rebuilt lazily with ``dist.new_group``.
        options: see :class:`ReplicationOptions`.

    .. warning::
        With ``election="group"`` this class issues a collective from inside
        ``prepare_local_plan``. If another rank raises earlier in local planning
        (a missing key or a shape mismatch, both of which DCP reports per rank)
        it will never reach that collective and its peers will block. Prefer
        the default ``"coordinator"`` unless you are running ``no_dist=True``,
        and give the replication group a bounded ``timeout`` if you do.
    """

    def __init__(
        self,
        storage_reader: StorageReader,
        *,
        replication_group: Optional[dist.ProcessGroup] = None,
        replication_ranks: Optional[list[list[int]]] = None,
        options: Optional[ReplicationOptions] = None,
    ) -> None:
        if replication_group is not None and replication_ranks is not None:
            raise ValueError(
                "pass replication_group or replication_ranks, not both"
            )
        self.storage_reader = storage_reader
        self.options = options or ReplicationOptions()
        self._group = replication_group
        self._partition = (
            tuple(tuple(r) for r in replication_ranks)
            if replication_ranks is not None
            else None
        )
        self._metadata: Optional[Metadata] = None
        self._local_items: list[ReadItem] = []
        self._local_schedule: Optional[_Schedule] = None
        self._exchange_device: Optional[torch.device] = None
        self._cache: dict[str, Any] = {}
        self._dcp_rank: Optional[int] = None

        # Observability, kept cheap so callers can assert on it in tests.
        self.num_broadcasts = 0
        self.plan_items_in = 0
        self.plan_items_out = 0
        self.bytes_exchanged = 0
        self.num_transfers = 0
        self.read_seconds = 0.0
        self.exchange_seconds = 0.0

    # -- plumbing ----------------------------------------------------------

    @property
    def group(self):
        if self._group is None and self._partition is not None:
            self._group = _resolve_group(self._partition)
        return self._group

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        return self.storage_reader.checkpoint_id

    def reset(self, checkpoint_id=None) -> None:
        self.storage_reader.reset(checkpoint_id)
        self._local_schedule = None
        self._local_items = []

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        return self.storage_reader.read_metadata(*args, **kwargs)

    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        self._metadata = metadata
        self._dcp_rank = kwargs.get("rank")
        self.storage_reader.set_up_storage_reader(
            metadata, is_coordinator, *args, **kwargs
        )

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id) -> bool:
        # Only consulted when DCP infers a reader from a checkpoint_id, which
        # cannot select a decorator.
        return False

    def _resolved_device(self) -> torch.device:
        if self._exchange_device is not None:
            return self._exchange_device
        device = self.options.exchange_device
        if device is None:
            group = self.group or dist.group.WORLD
            supported = getattr(group, "_device_types", [])
            if torch.cuda.is_available() and torch.device("cuda") in supported:
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
        self._exchange_device = device
        return device

    def _enabled(self) -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size(self.group) > 1

    # -- planning ----------------------------------------------------------

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        plan = self.storage_reader.prepare_local_plan(plan)
        self._local_items = list(plan.items)
        self.plan_items_in = len(plan.items)
        self.plan_items_out = len(plan.items)
        self._local_schedule = None

        if not self._enabled():
            return plan

        if self.options.election == "coordinator":
            ranks = tuple(dist.get_process_group_ranks(self.group))
            return dataclasses.replace(plan, storage_data=_Tag(plan.storage_data, ranks))

        return self._elect_within_group(plan)

    def _elect_within_group(self, plan: LoadPlan) -> LoadPlan:
        if self._metadata is None:
            raise AssertionError("set_up_storage_reader must run before prepare_local_plan")
        group = self.group
        group_size = dist.get_world_size(group)
        group_ranks = tuple(dist.get_process_group_ranks(group))
        my_rank = dist.get_rank(group)

        keys = [
            _read_key(i) for i in plan.items if i.type is not LoadItemType.BYTE_IO
        ]
        owners, entries = self._group_election(keys, group, group_size, group_ranks)

        self._local_schedule = _Schedule(plan.storage_data, entries)
        keep = [
            i
            for i in plan.items
            if i.type is LoadItemType.BYTE_IO or owners[_read_key(i)] == my_rank
        ]
        self.plan_items_out = len(keep)
        return dataclasses.replace(plan, items=keep)

    def _group_election(self, keys, group, group_size, group_ranks):
        """Elect owners inside the replication group.

        Shipping every rank's key list was the second largest cost in a load
        after the storage read -- 1.8 s of a 14 s 16B load, nearly all of it
        pickling -- and it is usually unnecessary. In a replication group every
        member wants the same keys, which is what the group *is*, and one
        fixed-size all_gather of digests establishes that. When the digests
        agree, ``needers`` is "all ranks" for every key, which each rank already
        knows, so no key data goes on the wire at all.
        """
        digests = _gather_digests(_digest(keys), group, group_size)

        cached = self._cache.get("group") if self.options.enable_plan_caching else None
        if cached is not None and cached[0] == digests:
            logger.debug("replication: reusing cached group election")
            return cached[1], cached[2]

        needers: dict[_ReadKey, list[int]] = {}
        if len(set(digests)) == 1:
            everyone = list(range(group_size))
            needers = dict.fromkeys(keys, everyone)
        else:
            # Not uniformly replicated: a resharding load, or a group that is
            # not really a replication group. Fall back to the key lists.
            logger.debug("replication: key sets differ across the group, gathering plans")
            gathered: list[list[_ReadKey]] = [None] * group_size  # type: ignore[list-item]
            dist.all_gather_object(gathered, keys, group=group)
            by_key: dict[_ReadKey, list[int]] = defaultdict(list)
            for rank, rank_keys in enumerate(gathered):
                for key in rank_keys:
                    by_key[key].append(rank)
            needers = by_key
        sizes = {k: _read_nbytes(k, self._metadata) for k in needers}
        owners = _elect_owners(needers, sizes, group_size, _sort_key)
        entries = [
            (group_ranks[owners[k]], k)
            for k in sorted(needers, key=_sort_key)
            if len(needers[k]) > 1
        ]

        if self.options.enable_plan_caching:
            self._cache["group"] = (digests, owners, entries)
        return owners, entries

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        if not self._enabled() or self.options.election != "coordinator":
            return self.storage_reader.prepare_global_plan(plans)
        if self._metadata is None:
            raise AssertionError("set_up_storage_reader must run before prepare_global_plan")

        plans = self.storage_reader.prepare_global_plan(plans)
        tags: list[_Tag] = [p.storage_data for p in plans]

        if len(plans) == 1 and dist.get_world_size(self.group) > 1:
            warnings.warn(
                "ReplicaAwareStorageReader saw a single load plan, which means "
                "dcp.load was called with no_dist=True. election='coordinator' "
                "cannot deduplicate there; pass election='group' instead.",
                stacklevel=2,
            )
            return [
                dataclasses.replace(p, storage_data=t.inner)
                for p, t in zip(plans, tags)
            ]

        group_of = {r: t.group_ranks for r, t in enumerate(tags)}
        per_group: dict[tuple[int, ...], dict[_ReadKey, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rank, plan in enumerate(plans):
            group = group_of[rank]
            local_rank = group.index(rank)
            for item in plan.items:
                if item.type is not LoadItemType.BYTE_IO:
                    per_group[group][_read_key(item)].append(local_rank)

        owners_of, schedule_of = self._coordinator_election(per_group)

        out: list[LoadPlan] = []
        for rank, plan in enumerate(plans):
            group = group_of[rank]
            owners = owners_of.get(group, {})
            keep = [
                i
                for i in plan.items
                if i.type is LoadItemType.BYTE_IO
                or group[owners[_read_key(i)]] == rank
            ]
            out.append(
                dataclasses.replace(
                    plan,
                    items=keep,
                    storage_data=_Schedule(tags[rank].inner, schedule_of[group]),
                )
            )
        return out

    def _coordinator_election(self, per_group):
        # The signature has to cover *which ranks* need each key, not just the
        # key set: the same keys spread over different ranks is a different
        # election.
        signature = {
            group: _fingerprint(
                "|".join(
                    f"{_sort_key(k)}:{sorted(ranks)}"
                    for k, ranks in sorted(needers.items(), key=lambda kv: _sort_key(kv[0]))
                )
            )
            for group, needers in per_group.items()
        }
        cached = self._cache.get("coordinator") if self.options.enable_plan_caching else None
        if cached is not None and cached[0] == signature:
            logger.debug("replication: reusing cached coordinator election")
            return cached[1], cached[2]

        owners_of: dict[tuple[int, ...], dict[_ReadKey, int]] = {}
        schedule_of: dict[tuple[int, ...], list[tuple[int, _ReadKey]]] = {}
        for group, needers in per_group.items():
            sizes = {k: _read_nbytes(k, self._metadata) for k in needers}
            owners = _elect_owners(needers, sizes, len(group), _sort_key)
            owners_of[group] = owners
            schedule_of[group] = [
                (group[owners[k]], k)
                for k in sorted(needers, key=_sort_key)
                if len(needers[k]) > 1
            ]
        if self.options.enable_plan_caching:
            self._cache["coordinator"] = (signature, owners_of, schedule_of)
        return owners_of, schedule_of

    # -- reading -----------------------------------------------------------

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        schedule = self._schedule_for(plan)
        if schedule is None:
            return self.storage_reader.read_data(plan, planner)

        me = dist.get_rank()
        to_send = {key for src, key in schedule.entries if src == me}
        device = self._resolved_device()

        capture = (
            _CapturingPlanner(planner, to_send, device) if to_send else None
        )
        inner_plan = dataclasses.replace(plan, storage_data=schedule.inner)
        started = time.perf_counter()
        self.storage_reader.read_data(inner_plan, capture or planner).wait()
        self.read_seconds += time.perf_counter() - started

        if schedule.entries:
            started = time.perf_counter()
            self._exchange(
                schedule, plan, planner, capture.captured if capture else {}, device
            )
            # The broadcasts and the copies into the destinations are enqueued,
            # not completed, so without this the timer reports launch cost and
            # makes the exchange look free. Correctness does not need it: the
            # work is ordered on the stream the caller goes on to use.
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.exchange_seconds += time.perf_counter() - started

        logger.info(
            "replica-aware load: storage %.2fs, exchange %.2fs over %d broadcast(s) "
            "and %d rdma read(s) moving %.1f MiB "
            "(transport=%s bucket_bytes=%d)",
            self.read_seconds,
            self.exchange_seconds,
            self.num_broadcasts,
            self.num_transfers,
            self.bytes_exchanged / 2**20,
            self.options.transport,
            self.options.bucket_bytes,
        )

        result: Future = Future()
        result.set_result(None)
        return result

    def _schedule_for(self, plan: LoadPlan) -> Optional[_Schedule]:
        if not self._enabled():
            return None
        if self.options.election == "group":
            return self._local_schedule
        return plan.storage_data if isinstance(plan.storage_data, _Schedule) else None

    def _exchange(
        self,
        schedule: _Schedule,
        plan: LoadPlan,
        planner: LoadPlanner,
        captured: dict[_ReadKey, torch.Tensor],
        device: torch.device,
    ) -> None:
        if self._metadata is None:
            raise AssertionError("metadata is not set up")

        read_here = {
            _read_key(i) for i in plan.items if i.type is not LoadItemType.BYTE_IO
        }
        needed: dict[_ReadKey, list[ReadItem]] = defaultdict(list)
        for item in self._local_items:
            if item.type is LoadItemType.BYTE_IO:
                continue
            key = _read_key(item)
            if key not in read_here:
                needed[key].append(item)

        filled: set[_ReadKey] = set()
        if self.options.transport == "nixl":
            exchange = self._exchange_nixl
        elif self.options.bucket_bytes > 0:
            exchange = self._exchange_bucketed
        else:
            exchange = self._exchange_per_key
        exchange(schedule, planner, captured, needed, device, filled)

        if self.options.validate:
            missing = set(needed) - filled
            if missing:
                raise RuntimeError(
                    f"replica-aware load did not satisfy {len(missing)} read "
                    f"request(s), e.g. {sorted(missing, key=_sort_key)[0][0].fqn}. "
                    "This usually means the replication group does not match the "
                    "actual data replication."
                )

    def _fill(self, key, buffer, planner, needed, filled) -> None:
        for item in needed.get(key, ()):
            target = planner.resolve_tensor(item).detach()
            target.copy_(buffer)
            planner.commit_tensor(item, target)
            filled.add(key)

    def _exchange_per_key(
        self, schedule, planner, captured, needed, device, filled
    ) -> None:
        me = dist.get_rank()
        group = self.group
        for src, key in schedule.entries:
            if src == me:
                buffer = captured[key]
            else:
                # Receivers and bystanders both have to join the collective;
                # bystanders discard their buffer.
                buffer = torch.empty(
                    tuple(key[2]),
                    dtype=_tensor_md(key, self._metadata).properties.dtype,
                    device=device,
                )
            dist.broadcast(buffer, src=src, group=group)
            self.num_broadcasts += 1
            self.bytes_exchanged += buffer.numel() * buffer.element_size()
            self._fill(key, buffer, planner, needed, filled)

    def _exchange_bucketed(
        self, schedule, planner, captured, needed, device, filled
    ) -> None:
        me = dist.get_rank()
        group = self.group
        for src, dtype, keys in _bucket(
            schedule.entries, self._metadata, self.options.bucket_bytes
        ):
            counts = [math.prod(k[2]) for k in keys]
            flat = torch.empty(sum(counts), dtype=dtype, device=device)
            if src == me:
                offset = 0
                for key, count in zip(keys, counts):
                    flat[offset : offset + count].copy_(captured[key].reshape(-1))
                    offset += count
            dist.broadcast(flat, src=src, group=group)
            self.num_broadcasts += 1
            self.bytes_exchanged += flat.numel() * flat.element_size()

            offset = 0
            for key, count in zip(keys, counts):
                buffer = flat[offset : offset + count].view(tuple(key[2]))
                offset += count
                self._fill(key, buffer, planner, needed, filled)

    def _exchange_nixl(
        self, schedule, planner, captured, needed, device, filled
    ) -> None:
        arena = _nixl_arena(self.group, device, self.options.arena_bytes)
        me = dist.get_rank()

        # A chunk larger than one arena slot cannot be staged: _waves gives it a
        # wave to itself, but writing it would run past the slot into the next
        # rank's, which is silent corruption rather than an error. Refuse up
        # front and say what to set.
        largest = max(
            (_read_nbytes(key, self._metadata) for _src, key in schedule.entries),
            default=0,
        )
        if largest > arena.wave_bytes:
            raise ValueError(
                f"arena_bytes={self.options.arena_bytes} gives {arena.wave_bytes} "
                f"bytes per group member, too small for a {largest}-byte chunk. "
                f"Set arena_bytes to at least "
                f"{largest * dist.get_world_size(self.group)}."
            )

        for wave in _waves(schedule.entries, self._metadata, arena.wave_bytes):
            for key, offset, nbytes in wave.get(me, ()):
                dtype = _tensor_md(key, self._metadata).properties.dtype
                arena.window(me, offset, nbytes, dtype).copy_(
                    captured[key].reshape(-1)
                )
            # A collective gave the "your peer's buffer is ready" edge for free;
            # a one-sided read does not, so the arenas are fenced explicitly.
            # Two barriers per wave replace one collective per chunk.
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier(group=self.group)

            posted = []
            for src, packed in wave.items():
                if src == me:
                    continue
                wanted = [(o, n) for key, o, n in packed if key in needed]
                if wanted:
                    posted.append(arena.post_read(src, wanted))
                    self.num_transfers += 1
            arena.await_reads(posted)

            for src, packed in wave.items():
                if src == me:
                    continue
                for key, offset, nbytes in packed:
                    if key not in needed:
                        continue
                    dtype = _tensor_md(key, self._metadata).properties.dtype
                    buffer = arena.window(src, offset, nbytes, dtype)
                    self._fill(
                        key, buffer.view(tuple(key[2])), planner, needed, filled
                    )
                    self.bytes_exchanged += nbytes

            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier(group=self.group)


def _load_nixl():
    """Import NIXL from whichever wheel is installed, or raise with the fix."""
    import importlib

    for package in ("nixl_cu13", "nixl_cu12", "nixl"):
        try:
            return importlib.import_module(f"{package}._api")
        except ModuleNotFoundError as exc:
            if exc.name not in (package, f"{package}._api"):
                raise
    raise RuntimeError(
        "transport='nixl' needs the nixl package: pip install nixl. "
        "Use transport='broadcast' to stay on torch.distributed."
    )


# One agent and one registration per (group, device); creating them is far more
# expensive than a load, and a second agent on the same device would re-pay
# ibv_reg_mr for nothing.
_NIXL_CONTEXTS: dict[tuple, "_NixlArena"] = {}


def _rdma_buffer(nbytes: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Allocate device memory suitable for RDMA registration, and a view of it.

    Deliberately not ``torch.empty``. Under
    ``PYTORCH_ALLOC_CONF=expandable_segments:True`` -- which torchtitan,
    Megatron-LM and most large training setups enable -- the caching allocator
    backs a tensor with a ``cuMemCreate``/``cuMemMap`` range rather than a
    ``cudaMalloc`` block, and remaps its physical pages as the segment grows.
    Legacy CUDA IPC is not valid on VMM allocations, so registering such a
    tensor appears to succeed and then faults on the first transfer, inside
    ``uct_cuda_ipc_ep_get_zcopy``. That is what killed the two-node run in job
    6897702; job 6913196 reproduced it by toggling this one setting and nothing
    else. ``cudaMalloc`` gives a mapping that stays put and that both the IPC
    and the network paths accept.

    Returns the view and the raw pointer, which the caller keeps for ``cudaFree``.
    """
    from cuda.bindings import runtime as cudart

    from torch.cuda._utils import _check_cuda_bindings

    with torch.cuda.device(device):
        ptr = int(_check_cuda_bindings(cudart.cudaMalloc(nbytes)))

    # torch reads __cuda_array_interface__ off the object with getattr, so an
    # instance attribute is enough and no class is needed. The tensor shares the
    # region rather than copying it.
    region = types.SimpleNamespace(
        __cuda_array_interface__={
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "strides": None,
            "version": 2,
        }
    )
    return torch.as_tensor(region, device=device), ptr


class _NixlArena:
    """A NIXL agent plus one registered staging arena shared by a group.

    The arena is ``group_size`` equal slots of ``wave_bytes``. Slot ``i`` holds
    what rank ``i`` published, on every rank: a source packs into its own slot,
    and a reader pulls a peer's slot into the identically placed slot of its own
    arena. Because the layout matches, a read needs only the peer's base
    pointer, which is one fixed-size all_gather of int64 rather than an address
    table per chunk.
    """

    def __init__(self, group, device: torch.device, arena_bytes: int) -> None:
        api = _load_nixl()
        ranks = tuple(dist.get_process_group_ranks(group))
        size = len(ranks)
        self.group = group
        self.slot_of = {rank: i for i, rank in enumerate(ranks)}
        self.wave_bytes = max(_ALIGN, (arena_bytes // size) // _ALIGN * _ALIGN)
        self.mem = "VRAM" if device.type == "cuda" else "DRAM"
        self.dev_id = device.index if device.index is not None else 0
        # UCX is the portable default, but a deployment whose UCX conflicts with
        # the one NIXL ships can point this at another plugin (LIBFABRIC, ...).
        self.backends = [os.environ.get("DCP_NIXL_BACKEND", "UCX")]

        me = dist.get_rank()
        # Containers tuned for MPI pin UCX to one transport -- the NGC pytorch
        # image sets UCX_TLS=tcp -- and NIXL's UCX backend refuses to start on
        # that: "Invalid UCX_TLS=tcp for NIXL UCX backend: NVIDIA GPU(s) are
        # present, but this setting does not enable CUDA memory support." So
        # override it while the agent is created and restore it after.
        #
        # The list must name an RDMA transport explicitly. UCX only considers
        # what UCX_TLS allows, so a GPU-only list ("cuda", which expands to
        # cuda_copy and cuda_ipc, plus the same-node CPU paths) leaves tcp as
        # the only thing that reaches another node, and cross-node reads fall
        # back to software emulation over the management NIC. On a 16B
        # checkpoint that was 126.93s of exchange; adding rc took the identical
        # transfer to 2.10s.
        #
        # UCX_NET_DEVICES is deliberately NOT touched. The image lists the HCAs
        # that are actually cabled; clearing it lets UCX choose one that is not
        # (mlx5_5 here) and the first transfer dies in ibv_create_ah with a
        # connect timeout.
        previous_tls = os.environ.get("UCX_TLS")
        os.environ["UCX_TLS"] = os.environ.get(
            "DCP_NIXL_UCX_TLS", "cuda_copy,cuda_ipc,rc,tcp,self,sm"
        )
        try:
            self.agent = api.nixl_agent(
                f"dcp-replica-{me}", api.nixl_agent_config(backends=self.backends)
            )
        finally:
            if previous_tls is None:
                os.environ.pop("UCX_TLS", None)
            else:
                os.environ["UCX_TLS"] = previous_tls
        # Expandable segments are a CUDA-allocator feature, so only the device
        # arena has to sidestep the caching allocator.
        if device.type == "cuda":
            self.buffer, self.base = _rdma_buffer(self.wave_bytes * size, device)
        else:
            self.buffer = torch.empty(
                self.wave_bytes * size, dtype=torch.uint8, device=device
            )
            self.base = self.buffer.data_ptr()
        self.registration = self.agent.register_memory(
            self.buffer, backends=self.backends
        )

        # Base pointer and device index per rank: fixed shape, no pickling, no
        # object collective. The device index has to travel too -- a remote
        # descriptor names the memory by (address, size, device), and a peer's
        # arena is on the peer's GPU, not on ours. Getting that wrong is not a
        # wrong-data bug, it is NIXL_ERR_NOT_FOUND at prep time, because the
        # triple matches no region the peer registered.
        info = torch.zeros(size, 2, dtype=torch.int64, device=device)
        info[self.slot_of[me], 0] = self.base
        info[self.slot_of[me], 1] = self.dev_id
        dist.all_reduce(info, group=group)
        self.peer_base = {rank: int(info[i, 0]) for rank, i in self.slot_of.items()}
        self.peer_dev = {rank: int(info[i, 1]) for rank, i in self.slot_of.items()}

        # Agent metadata is the one variable-length payload. The rendezvous store
        # moves it point to point, so it needs no collective and no listener.
        store = dist.distributed_c10d._get_default_store()
        prefix = f"dcp_nixl/{'-'.join(map(str, ranks))}"
        store.set(f"{prefix}/{me}", self.agent.get_agent_metadata())
        self.peer_name = {}
        for rank in ranks:
            if rank != me:
                self.peer_name[rank] = self.agent.add_remote_agent(
                    store.get(f"{prefix}/{rank}")
                )

    def window(self, owner: int, offset: int, nbytes: int, dtype) -> torch.Tensor:
        start = self.slot_of[owner] * self.wave_bytes + offset
        return self.buffer[start : start + nbytes].view(dtype)

    def post_read(self, owner: int, wanted: list[tuple[int, int]]):
        """Start one batched READ of a peer's slot. Returns a handle to await."""
        slot = self.slot_of[owner] * self.wave_bytes
        remote = [
            (self.peer_base[owner] + slot + o, n, self.peer_dev[owner])
            for o, n in wanted
        ]
        local = [(self.base + slot + o, n, self.dev_id) for o, n in wanted]
        source = self.agent.prep_xfer_dlist(
            self.peer_name[owner], remote, mem_type=self.mem, backends=self.backends
        )
        dest = self.agent.prep_xfer_dlist(
            "", local, mem_type=self.mem, backends=self.backends
        )
        index = list(range(len(wanted)))
        handle = self.agent.make_prepped_xfer(
            "READ", dest, index, source, index, b"", self.backends
        )
        if self.agent.transfer(handle) == "ERR":
            self.agent.release_xfer_handle(handle)
            raise RuntimeError(f"NIXL READ from rank {owner} failed to post")
        return handle

    def await_reads(self, handles, timeout: float = 600.0) -> None:
        deadline = time.perf_counter() + timeout
        try:
            for handle in handles:
                while True:
                    state = self.agent.check_xfer_state(handle)
                    if state == "DONE":
                        break
                    if state == "ERR":
                        raise RuntimeError("NIXL READ failed in flight")
                    if time.perf_counter() > deadline:
                        raise RuntimeError("NIXL READ timed out")
        finally:
            for handle in handles:
                self.agent.release_xfer_handle(handle)


def _nixl_arena(group, device: torch.device, arena_bytes: int) -> _NixlArena:
    key = (tuple(dist.get_process_group_ranks(group)), str(device), arena_bytes)
    if key not in _NIXL_CONTEXTS:
        _NIXL_CONTEXTS[key] = _NixlArena(group, device, arena_bytes)
    return _NIXL_CONTEXTS[key]


class ReplicaAwareStorageWriter(StorageWriter):
    """Drop replicated writes before the plan leaves the rank.

    EXPERIMENTAL.

    DCP already deduplicates writes, but only on the coordinator, after every
    rank has shipped its full ``SavePlan`` to rank 0. That gather is
    O(world_size x items_per_rank). This decorator runs a first deduplication
    pass inside the replication group, so the plan that enters the gather is
    roughly ``group_size`` times smaller.

    It is a pure optimisation: DCP's global ``dedup_save_plans`` still runs and
    remains authoritative, so a duplicate this pass cannot see (the same tensor
    replicated across two different groups) is still removed. The invariant we
    must not break is "never drop an item on every rank", and electing exactly
    one owner out of the ranks that proposed it guarantees a keeper.

    Args:
        storage_writer: the writer to decorate.
        replication_group: the process group whose members hold replicas of each
            other, typically the data-parallel group.
        replication_ranks: alternative for callers that cannot hold a live
            ``ProcessGroup``; a ``ProcessGroup`` is not picklable, so this is
            what makes ``AsyncCheckpointerType.PROCESS`` work. Give the full
            partition of the world, e.g. ``[[0, 2], [1, 3]]``.
        options: see :class:`ReplicationOptions`. Only ``validate`` and
            ``enable_plan_caching`` apply to the write path.

    .. warning::
        Incompatible with ``dcp.save(..., use_collectives=False)``, which makes
        every rank write a ``__{rank}.metadata`` describing only its own writes.
        Deduplicating across ranks makes those files non-self-contained and the
        checkpoint fails to reload, so this combination raises.
    """

    def __init__(
        self,
        storage_writer: StorageWriter,
        *,
        replication_group: Optional[dist.ProcessGroup] = None,
        replication_ranks: Optional[list[list[int]]] = None,
        options: Optional[ReplicationOptions] = None,
    ) -> None:
        if replication_group is not None and replication_ranks is not None:
            raise ValueError("pass replication_group or replication_ranks, not both")
        self.storage_writer = storage_writer
        self.options = options or ReplicationOptions()
        self._group = replication_group
        self._partition = (
            tuple(tuple(r) for r in replication_ranks)
            if replication_ranks is not None
            else None
        )
        self._cache: dict[str, Any] = {}

        self.plan_items_in = 0
        self.plan_items_out = 0

    @property
    def group(self):
        if self._group is None and self._partition is not None:
            self._group = _resolve_group(self._partition)
        return self._group

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        return self.storage_writer.checkpoint_id

    def _enabled(self) -> bool:
        return (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size(self.group) > 1
        )

    def reset(self, checkpoint_id=None) -> None:
        self.storage_writer.reset(checkpoint_id)

    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        # Resolving the group here rather than lazily puts its `new_group`
        # collective at a point every rank reaches symmetrically.
        if not kwargs.get("use_collectives", True) and self._enabled():
            raise RuntimeError(
                "ReplicaAwareStorageWriter cannot be combined with "
                "use_collectives=False: without a coordinator each rank writes a "
                "__{rank}.metadata covering only its own writes, and "
                "deduplicating across ranks makes those files incomplete, so the "
                "checkpoint will not reload. Use the default use_collectives=True."
            )
        self.storage_writer.set_up_storage_writer(is_coordinator, *args, **kwargs)

    def storage_meta(self):
        return self.storage_writer.storage_meta()

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.plan_items_in = len(plan.items)
        if self._enabled():
            plan = dataclasses.replace(plan, items=self._elect(plan.items))
        self.plan_items_out = len(plan.items)
        return self.storage_writer.prepare_local_plan(plan)

    def _elect(self, items: list[WriteItem]) -> list[WriteItem]:
        """Keep only the items this rank was elected to write.

        Runs unconditionally when enabled, even for an empty plan. With
        `enable_plan_caching` a rank whose plan is unchanged contributes
        `SavePlan([], usable=False)`, and if only *some* ranks skipped the
        collective the rest would block in it forever.
        """
        group = self.group
        group_size = dist.get_world_size(group)
        my_rank = dist.get_rank(group)
        payload = [(item.index, _write_nbytes(item)) for item in items]

        digests = _gather_digests(_payload_digest(payload), group, group_size)

        cached = self._cache.get("save") if self.options.enable_plan_caching else None
        if cached is not None and cached[0] == digests:
            logger.debug("replication: reusing cached write election")
            return [i for i in items if cached[1][i.index] == my_rank]

        needers: dict[MetadataIndex, list[int]] = {}
        sizes: dict[MetadataIndex, int] = {}
        if len(set(digests)) == 1:
            # Every rank proposes the same writes, so each can derive the same
            # election without anyone shipping a plan. Note this is also the
            # case a rank with an unusable cached plan does NOT hit: its payload
            # is empty, its digest differs, and the gather below still runs.
            everyone = list(range(group_size))
            needers = {index: everyone for index, _ in payload}
            sizes = dict(payload)
        else:
            gathered: list[list[tuple[MetadataIndex, int]]] = [None] * group_size  # type: ignore[list-item]
            dist.all_gather_object(gathered, payload, group=group)
            by_index: dict[MetadataIndex, list[int]] = defaultdict(list)
            for rank, rank_payload in enumerate(gathered):
                for index, nbytes in rank_payload:
                    by_index[index].append(rank)
                    sizes.setdefault(index, nbytes)
            needers = by_index
        owners = _elect_owners(needers, sizes, group_size, _write_sort_key)

        if self.options.enable_plan_caching:
            self._cache["save"] = (digests, owners)
        return [i for i in items if owners[i.index] == my_rank]

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        return self.storage_writer.prepare_global_plan(plans)

    def write_data(self, plan: SavePlan, planner: SavePlanner):
        return self.storage_writer.write_data(plan, planner)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        self.storage_writer.finish(metadata, results)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id) -> bool:
        return False
