# Owner(s): ["oncall: distributed_checkpointing"]

import contextlib
import importlib.util
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import filesystem as dcp_filesystem
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.replication import (
    _ALIGN,
    _elect_owners,
    _load_nixl,
    _rdma_buffer,
    _read_key,
    _sort_key,
    _waves,
    ReplicaAwareStorageReader,
    ReplicaAwareStorageWriter,
    ReplicationOptions,
)
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def _tensor_key(fqn, offsets, lengths):
    return (
        MetadataIndex(fqn, torch.Size(offsets)),
        torch.Size(offsets),
        torch.Size(lengths),
    )


def _sizes(needers):
    """Bytes per key, derived from its lengths, for the election tests."""
    out = {}
    for key in needers:
        numel = 1
        for size in key[2]:
            numel *= size
        out[key] = numel * 4
    return out


def _fake_metadata(keys, dtype=torch.float32):
    md = {}
    for key in keys:
        fqn = key[0].fqn
        md.setdefault(
            fqn,
            TensorStorageMetadata(
                properties=TensorProperties(dtype=dtype),
                size=torch.Size([1024, 1024]),
                chunks=[],
            ),
        )
        md[fqn].chunks.append(ChunkStorageMetadata(offsets=key[0].offset, sizes=key[2]))
    return Metadata(state_dict_metadata=md)


@contextlib.contextmanager
def _count_read_bytes():
    """Count bytes pulled through FileSystem read streams on this rank."""
    counter = {"read": 0}
    original = dcp_filesystem.FileSystem.create_stream

    class _Counting:
        def __init__(self, stream):
            self._stream = stream

        def read(self, size=-1):
            data = self._stream.read(size)
            counter["read"] += len(data)
            return data

        def readinto(self, b):
            n = self._stream.readinto(b)
            counter["read"] += n or 0
            return n

        def __getattr__(self, name):
            return getattr(self._stream, name)

    @contextlib.contextmanager
    def create_stream(self, path, mode):
        with original(self, path, mode) as stream:
            yield _Counting(stream) if "r" in mode else stream

    dcp_filesystem.FileSystem.create_stream = create_stream
    try:
        yield counter
    finally:
        dcp_filesystem.FileSystem.create_stream = original


class TestReplicationElection(TestCase):
    """Pure-python properties of the election. No process group needed."""

    def test_election_is_deterministic(self):
        keys = [_tensor_key(f"w{i}", [i, 0], [4, 4]) for i in range(16)]
        needers = {k: [0, 1, 2, 3] for k in keys}

        reference = _elect_owners(needers, _sizes(needers), 4, _sort_key)
        for _ in range(50):
            shuffled = {k: list(needers[k]) for k in reversed(keys)}
            self.assertEqual(
                _elect_owners(shuffled, _sizes(shuffled), 4, _sort_key), reference
            )

    def test_election_never_drops_and_stays_in_the_replica_set(self):
        keys = [_tensor_key(f"w{i}", [i, 0], [4, 4]) for i in range(12)]
        needers = {k: sorted({(i * 7) % 4, (i * 3) % 4}) for i, k in enumerate(keys)}

        owners = _elect_owners(needers, _sizes(needers), 4, _sort_key)
        self.assertEqual(set(owners), set(keys))
        for key, owner in owners.items():
            self.assertIn(owner, needers[key])

    def test_election_balances_uniform_replicas(self):
        keys = [_tensor_key(f"w{i}", [i, 0], [8, 8]) for i in range(8)]
        owners = _elect_owners(
            {k: [0, 1, 2, 3] for k in keys}, _sizes({k: 0 for k in keys}), 4, _sort_key
        )

        per_rank = [0] * 4
        for owner in owners.values():
            per_rank[owner] += 1
        self.assertEqual(per_rank, [2, 2, 2, 2])

    def test_election_accounts_for_single_rank_keys(self):
        """Keys with one candidate still consume that rank's budget.

        They are dropped from the *broadcast schedule* (nobody to send them to)
        but never from the election input, so the balancer knows rank 0 is
        already busy and gives it less of the shared work.
        """
        big = _tensor_key("big", [0, 0], [64, 64])
        shared = [_tensor_key(f"s{i}", [i, 0], [8, 8]) for i in range(8)]
        needers = {big: [0]}
        needers.update({k: [0, 1, 2, 3] for k in shared})

        owners = _elect_owners(needers, _sizes(needers), 4, _sort_key)

        self.assertEqual(owners[big], 0)  # only candidate
        per_rank = [0] * 4
        for key in shared:
            per_rank[owners[key]] += 1
        self.assertEqual(sum(per_rank), len(shared))
        self.assertLess(per_rank[0], per_rank[1])

    def test_read_key_ignores_the_destination(self):
        def item(dest_offsets, storage_offsets, lengths):
            return ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex("w", torch.Size(dest_offsets)),
                dest_offsets=torch.Size(dest_offsets),
                storage_index=MetadataIndex("w", torch.Size(storage_offsets)),
                storage_offsets=torch.Size(storage_offsets),
                lengths=torch.Size(lengths),
            )

        base = item([0, 0], [0, 0], [4, 4])
        self.assertEqual(_read_key(base), _read_key(item([8, 8], [0, 0], [4, 4])))
        self.assertNotEqual(_read_key(base), _read_key(item([0, 0], [4, 0], [4, 4])))
        self.assertNotEqual(_read_key(base), _read_key(item([0, 0], [0, 0], [2, 4])))

    def _wave_metadata(self, keys, dtype=torch.float32):
        return Metadata(
            state_dict_metadata={
                key[0].fqn: TensorStorageMetadata(
                    properties=TensorProperties(dtype=dtype),
                    size=torch.Size(key[2]),
                    chunks=[
                        ChunkStorageMetadata(
                            offsets=torch.Size([0] * len(key[2])),
                            sizes=torch.Size(key[2]),
                        )
                    ],
                )
                for key in keys
            }
        )

    def test_waves_pack_per_source_with_aligned_offsets(self):
        keys = [_tensor_key(f"w{i}", [0], [16]) for i in range(6)]
        md = self._wave_metadata(keys)
        entries = [(i % 2, keys[i]) for i in range(6)]

        waves = list(_waves(entries, md, 1 << 20))
        self.assertEqual(len(waves), 1)
        self.assertEqual(set(waves[0]), {0, 1})
        for packed in waves[0].values():
            self.assertEqual([o for _k, o, _n in packed], [0, 64, 128])
            for _k, offset, _n in packed:
                self.assertEqual(offset % _ALIGN, 0)

    def test_waves_split_on_the_slot_and_lose_nothing(self):
        keys = [_tensor_key(f"w{i}", [0], [16]) for i in range(6)]
        md = self._wave_metadata(keys)
        entries = [(i % 2, keys[i]) for i in range(6)]

        waves = list(_waves(entries, md, 128))
        self.assertEqual(len(waves), 2)
        seen = [k for w in waves for packed in w.values() for k, _o, _n in packed]
        self.assertEqual(sorted(seen, key=_sort_key), sorted(keys, key=_sort_key))
        for wave in waves:
            for packed in wave.values():
                total = max(o + n for _k, o, n in packed)
                self.assertLessEqual(total, 128)

    def test_waves_carry_a_chunk_larger_than_a_slot(self):
        key = _tensor_key("big", [0], [1024])
        md = self._wave_metadata([key])
        waves = list(_waves([(0, key)], md, 8))
        self.assertEqual(len(waves), 1)
        self.assertEqual(len(waves[0][0]), 1)

    def test_waves_are_deterministic_and_handle_an_empty_schedule(self):
        keys = [_tensor_key(f"w{i}", [0], [16]) for i in range(6)]
        md = self._wave_metadata(keys)
        entries = [(i % 2, keys[i]) for i in range(6)]
        self.assertEqual(list(_waves(entries, md, 128)), list(_waves(entries, md, 128)))
        self.assertEqual(list(_waves([], md, 128)), [])

    def test_rejects_bad_transport_options(self):
        self.assertEqual(ReplicationOptions().transport, "broadcast")
        self.assertEqual(ReplicationOptions().bucket_bytes, 0)
        with self.assertRaises(ValueError):
            ReplicationOptions(transport="rdma")
        with self.assertRaises(ValueError):
            ReplicationOptions(arena_bytes=4)
        with self.assertRaises(ValueError):
            ReplicationOptions(bucket_bytes=-1)

    def test_missing_nixl_says_what_to_install(self):
        # A None entry in sys.modules makes the import raise ModuleNotFoundError,
        # so this exercises the missing-package path whether or not nixl is
        # actually installed.
        absent = {}
        for package in ("nixl_cu13", "nixl_cu12", "nixl"):
            absent[package] = None
            absent[f"{package}._api"] = None
        with mock.patch.dict(sys.modules, absent):
            with self.assertRaisesRegex(RuntimeError, "pip install nixl"):
                _load_nixl()

    def test_rejects_bad_options(self):
        with self.assertRaises(ValueError):
            ReplicationOptions(election="nope")
        with self.assertRaises(NotImplementedError):
            ReplicationOptions(dedup_bytes=True)

    @unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
    @unittest.skipIf(
        importlib.util.find_spec("cuda") is None
        or importlib.util.find_spec("cuda.bindings") is None,
        "needs cuda-python",
    )
    def test_rdma_buffer_survives_expandable_segments(self):
        # The arena is registered for one-sided RDMA, and legacy CUDA IPC is not
        # valid on the cuMemMap ranges the caching allocator hands out under
        # expandable_segments. Registering one appears to work and then faults
        # on the first transfer, which is what made a two-node NIXL load
        # segfault with no Python traceback. The buffer must therefore not come
        # from the caching allocator at all -- assert that directly, since the
        # symptom only shows up with a peer and a fabric attached.
        device = torch.device("cuda", torch.cuda.current_device())
        nbytes = 1 << 20
        before = torch.cuda.memory_allocated(device)
        buffer, ptr = _rdma_buffer(nbytes, device)
        try:
            self.assertEqual(buffer.numel(), nbytes)
            self.assertEqual(buffer.dtype, torch.uint8)
            self.assertEqual(buffer.data_ptr(), ptr)
            self.assertEqual(buffer.device.type, "cuda")
            # Nothing was taken from the caching allocator's pool.
            self.assertEqual(torch.cuda.memory_allocated(device), before)
            # And it is real, writable device memory that reads back.
            buffer.fill_(7)
            self.assertTrue(torch.all(buffer.cpu() == 7))
        finally:
            from cuda.bindings import runtime as cudart

            cudart.cudaFree(ptr)


class TestReplicaAwareLoad(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _tmpdir(self):
        path = tempfile.mkdtemp(prefix="dcp_repl_") if self.rank == 0 else None
        box = [path]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _state_dict(self, mesh, n=4, dim=256, shard_tp=True):
        sd = {}
        for i in range(n):
            full = (
                torch.arange(
                    dim * dim, dtype=torch.float32, device=self.device_type
                ).reshape(dim, dim)
                + i
            )
            placements = [Replicate(), Shard(0) if shard_tp else Replicate()]
            sd[f"layer{i}.weight"] = distribute_tensor(full, mesh, placements)
        return sd

    @staticmethod
    def _zeros_like(sd):
        return {
            k: DTensor.from_local(
                torch.zeros_like(v.to_local()),
                device_mesh=v.device_mesh,
                placements=v.placements,
                shape=v.size(),
                stride=v.stride(),
            )
            for k, v in sd.items()
        }

    def _mesh(self, dp, tp):
        return init_device_mesh(self.device_type, (dp, tp), mesh_dim_names=("dp", "tp"))

    def _reader(self, path, group, **kwargs):
        return ReplicaAwareStorageReader(
            FileSystemReader(path),
            replication_group=group,
            options=ReplicationOptions(**kwargs),
        )

    def _total_read(self, local_bytes):
        gathered = [0] * self.world_size
        dist.all_gather_object(gathered, local_bytes)
        return sum(gathered)

    def _assert_equal_sd(self, expected, actual):
        for k in expected:
            self.assertEqual(expected[k].to_local(), actual[k].to_local())

    # -- correctness -------------------------------------------------------

    @with_comms
    def test_load_matches_stock_and_reads_once(self):
        for dp in (1, 2, 4):
            tp = self.world_size // dp
            mesh = self._mesh(dp, tp)
            sd = self._state_dict(mesh)
            path = self._tmpdir()
            dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

            stock = self._zeros_like(sd)
            with _count_read_bytes() as counter:
                dcp.load(stock, storage_reader=FileSystemReader(path))
            stock_read = self._total_read(counter["read"])

            for election in ("coordinator", "group"):
                target = self._zeros_like(sd)
                reader = self._reader(path, mesh["dp"].get_group(), election=election)
                with _count_read_bytes() as counter:
                    dcp.load(target, storage_reader=reader)
                replica_read = self._total_read(counter["read"])

                self._assert_equal_sd(sd, target)
                self._assert_equal_sd(stock, target)
                # one reader per replica set: the saving is the group size
                self.assertAlmostEqual(stock_read / replica_read, dp, delta=0.05)
                if dp == 1:
                    self.assertEqual(reader.num_broadcasts, 0)

            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_resharding(self):
        cases = [
            ("sharded->replicated", (1, 4), (4, 1), 256, True, True),
            ("replicated->sharded", (4, 1), (1, 4), 256, True, True),
            ("identity", (1, 4), (1, 4), 256, True, True),
            ("uneven", (1, 4), (4, 1), 255, True, True),
            ("replicated both", (4, 1), (4, 1), 256, False, False),
        ]
        for name, save_mesh, load_mesh, dim, shard_save, shard_load in cases:
            sd = self._state_dict(self._mesh(*save_mesh), dim=dim, shard_tp=shard_save)
            reference = {k: v.full_tensor() for k, v in sd.items()}
            path = self._tmpdir()
            dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

            mesh = self._mesh(*load_mesh)
            target = self._zeros_like(
                self._state_dict(mesh, dim=dim, shard_tp=shard_load)
            )
            reader = self._reader(path, mesh["dp"].get_group())
            dcp.load(target, storage_reader=reader)

            for k, expected in reference.items():
                self.assertEqual(expected, target[k].full_tensor(), msg=name)

            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_subgroups_never_exchange_across_groups(self):
        mesh = self._mesh(2, 2)  # two replication groups of two
        sd = self._state_dict(mesh)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        target = self._zeros_like(sd)
        reader = self._reader(path, mesh["dp"].get_group())
        with _count_read_bytes() as counter:
            dcp.load(target, storage_reader=reader)

        self._assert_equal_sd(sd, target)
        # a chunk needed by both groups is read exactly once per group
        stock = self._zeros_like(sd)
        with _count_read_bytes() as stock_counter:
            dcp.load(stock, storage_reader=FileSystemReader(path))
        self.assertAlmostEqual(
            self._total_read(stock_counter["read"]) / self._total_read(counter["read"]),
            2.0,
            delta=0.05,
        )
        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_mixed_coverage_within_one_group(self):
        """A group where some keys are shared and some are wanted by one rank.

        This is the expert-parallel shape: inside a data-parallel group the
        non-expert weights are replicated while each rank owns a distinct expert
        shard. The shared key is broadcast once; the per-rank keys are read by
        their owner and never enter the schedule, because there is nobody to
        send them to.
        """
        mesh = self._mesh(self.world_size, 1)
        dim = 256
        shared = torch.arange(
            dim * dim, dtype=torch.float32, device=self.device_type
        ).reshape(dim, dim)
        unique = shared + 1
        sd = {
            "shared.weight": distribute_tensor(
                shared, mesh, [Replicate(), Replicate()]
            ),
            "unique.weight": distribute_tensor(unique, mesh, [Shard(0), Replicate()]),
        }
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        stock = self._zeros_like(sd)
        with _count_read_bytes() as counter:
            dcp.load(stock, storage_reader=FileSystemReader(path))
        stock_read = self._total_read(counter["read"])

        target = self._zeros_like(sd)
        reader = self._reader(path, mesh["dp"].get_group())
        with _count_read_bytes() as counter:
            dcp.load(target, storage_reader=reader)
        replica_read = self._total_read(counter["read"])

        self._assert_equal_sd(sd, target)
        self._assert_equal_sd(stock, target)
        # only the replicated tensor has more than one needer
        self.assertEqual(reader.num_broadcasts, 1)
        # Stock reads the shared tensor once per rank plus the sharded one once,
        # i.e. (world_size + 1) / 2 of the logical size; we read every byte once.
        # Bounds rather than equalities because each rank also reads `.metadata`.
        logical = 2 * dim * dim * 4
        self.assertGreater(stock_read / logical, (self.world_size + 1) / 2 - 0.05)
        self.assertLess(replica_read / logical, 1.2)
        self.assertGreater(replica_read / logical, 0.95)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_transforming_planner(self):
        """A planner that only moves data in commit_tensor must still work.

        Regression: reading the payload back via resolve_tensor broadcasts an
        empty scratch buffer.
        """
        calls = []

        class TransformingPlanner(DefaultLoadPlanner):
            def resolve_tensor(self, read_item):
                target = super().resolve_tensor(read_item)
                calls.append(("resolve", read_item.dest_index.fqn))
                self._pending = target
                return torch.empty_like(target)

            def commit_tensor(self, read_item, tensor):
                calls.append(("commit", read_item.dest_index.fqn))
                self._pending.copy_(tensor)
                self._pending = None

        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        target = self._zeros_like(sd)
        reader = self._reader(path, mesh["dp"].get_group())
        dcp.load(target, storage_reader=reader, planner=TransformingPlanner())

        self._assert_equal_sd(sd, target)
        # resolve/commit must stay strictly paired
        self.assertEqual(len(calls) % 2, 0)
        for i in range(0, len(calls), 2):
            self.assertEqual(calls[i][0], "resolve")
            self.assertEqual(calls[i + 1][0], "commit")
            self.assertEqual(calls[i][1], calls[i + 1][1])

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_dtype_comes_from_metadata(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            mesh = self._mesh(self.world_size, 1)
            full = torch.ones(128, 128, dtype=dtype, device=self.device_type)
            sd = {"w": distribute_tensor(full, mesh, [Replicate(), Replicate()])}
            path = self._tmpdir()
            dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

            target = self._zeros_like(sd)
            dcp.load(target, storage_reader=self._reader(path, mesh["dp"].get_group()))
            self.assertEqual(sd["w"].to_local(), target["w"].to_local())
            self.assertEqual(target["w"].to_local().dtype, dtype)

            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_no_dist_needs_group_election(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))
        group = mesh["dp"].get_group()

        target = self._zeros_like(sd)
        reader = self._reader(path, group, election="group")
        with _count_read_bytes() as counter:
            dcp.load(target, storage_reader=reader, no_dist=True)
        self._assert_equal_sd(sd, target)
        group_read = self._total_read(counter["read"])

        # coordinator election degenerates under no_dist, but stays correct
        target = self._zeros_like(sd)
        reader = self._reader(path, group, election="coordinator")
        with _count_read_bytes() as counter:
            dcp.load(target, storage_reader=reader, no_dist=True)
        self._assert_equal_sd(sd, target)
        self.assertGreater(self._total_read(counter["read"]), group_read)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_plan_caching_reuses_the_election(self):
        """Repeated loads stay correct and ship no key lists.

        This used to assert that the first load exchanged key lists and later
        ones only a checksum. Since the digest fast path landed, a uniformly
        replicated group ships no key data on any load. `no_dist=True` is what
        election="group" is for and is also what isolates our collectives:
        without it, DCP's own plan gather and scatter call all_gather_object
        and there is nothing of ours left to see.
        """
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        reader = self._reader(
            path, mesh["dp"].get_group(), election="group", enable_plan_caching=True
        )
        payloads = []
        original = dist.all_gather_object

        def counting(object_list, obj, group=None, **kwargs):
            payloads.append(obj)
            return original(object_list, obj, group=group, **kwargs)

        dist.all_gather_object = counting
        try:
            for _ in range(3):
                target = self._zeros_like(sd)
                dcp.load(target, storage_reader=reader, no_dist=True)
                self._assert_equal_sd(sd, target)
        finally:
            dist.all_gather_object = original

        self.assertEqual(payloads, [], f"expected no object collective, got {payloads}")

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_bucketing_loads_the_same_values_with_fewer_broadcasts(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh, n=8, dim=64)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        per_key = self._reader(path, mesh["dp"].get_group(), election="group")
        unbucketed = self._zeros_like(sd)
        dcp.load(unbucketed, storage_reader=per_key)
        self._assert_equal_sd(sd, unbucketed)

        # One bucket per source rank: 8 tensors of one dtype, limit above their total.
        bucketed = self._reader(
            path,
            mesh["dp"].get_group(),
            election="group",
            bucket_bytes=1 << 20,
        )
        target = self._zeros_like(sd)
        dcp.load(target, storage_reader=bucketed)

        self._assert_equal_sd(sd, target)
        self._assert_equal_sd(unbucketed, target)
        self.assertLess(bucketed.num_broadcasts, per_key.num_broadcasts)
        # Bucketing regroups the same bytes, it does not move different ones.
        self.assertEqual(bucketed.bytes_exchanged, per_key.bytes_exchanged)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_bucket_limit_is_respected_and_default_is_off(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh, n=8, dim=64)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        counts = {}
        for limit in (0, 1 << 13, 1 << 20):
            reader = self._reader(
                path, mesh["dp"].get_group(), election="group", bucket_bytes=limit
            )
            target = self._zeros_like(sd)
            dcp.load(target, storage_reader=reader)
            self._assert_equal_sd(sd, target)
            counts[limit] = reader.num_broadcasts

        # A tighter limit cuts more buckets, so it never issues fewer broadcasts.
        self.assertGreaterEqual(counts[1 << 13], counts[1 << 20])
        # bucket_bytes=0 is the untouched per-key path, so it is the most of all.
        self.assertGreaterEqual(counts[0], counts[1 << 13])
        self.assertEqual(ReplicationOptions().bucket_bytes, 0)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_bucketing_survives_resharding_and_mixed_dtypes(self):
        mesh_save = self._mesh(1, self.world_size)
        sd = self._state_dict(mesh_save, n=4, dim=64)
        sd["scalar"] = distribute_tensor(
            torch.arange(64, dtype=torch.bfloat16, device=self.device_type),
            mesh_save,
            [Replicate(), Replicate()],
        )
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        mesh_load = self._mesh(self.world_size, 1)
        expected = self._state_dict(mesh_load, n=4, dim=64)
        expected["scalar"] = distribute_tensor(
            torch.arange(64, dtype=torch.bfloat16, device=self.device_type),
            mesh_load,
            [Replicate(), Replicate()],
        )

        target = self._zeros_like(expected)
        reader = self._reader(
            path, mesh_load["dp"].get_group(), election="group", bucket_bytes=1 << 20
        )
        dcp.load(target, storage_reader=reader)
        self._assert_equal_sd(expected, target)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_uniform_group_ships_no_key_lists(self):
        """The point of the digest fast path: no object collective at all.

        Every member of a replication group wants the same keys, so the election
        can be derived locally once the digests agree. Gathering the key lists
        was 1.8s of a 14s 16B load, nearly all of it pickling.
        """
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh, n=8, dim=64)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        payloads = []
        original = dist.all_gather_object

        def counting(object_list, obj, group=None, **kwargs):
            payloads.append(obj)
            return original(object_list, obj, group=group, **kwargs)

        reader = self._reader(path, mesh["dp"].get_group(), election="group")
        target = self._zeros_like(sd)
        dist.all_gather_object = counting
        try:
            dcp.load(target, storage_reader=reader, no_dist=True)
        finally:
            dist.all_gather_object = original

        self._assert_equal_sd(sd, target)
        self.assertEqual(payloads, [], f"expected no object collective, got {payloads}")

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_non_uniform_group_falls_back_to_gathering_keys(self):
        """A group whose members want different keys must gather them.

        One tensor is sharded across the dp group, so each rank's key set
        differs, the digests disagree, and the election has to fall back to
        gathering the key lists instead of assuming everyone wants everything.
        """
        mesh = self._mesh(self.world_size, 1)
        dim = 64
        shared = torch.arange(
            dim * dim, dtype=torch.float32, device=self.device_type
        ).reshape(dim, dim)
        sd = {
            "shared.weight": distribute_tensor(
                shared, mesh, [Replicate(), Replicate()]
            ),
            "unique.weight": distribute_tensor(
                shared + 1, mesh, [Shard(0), Replicate()]
            ),
        }
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        gathered_key_lists = []
        original = dist.all_gather_object

        def counting(object_list, obj, group=None, **kwargs):
            if isinstance(obj, list):
                gathered_key_lists.append(obj)
            return original(object_list, obj, group=group, **kwargs)

        target = self._zeros_like(sd)
        reader = self._reader(path, mesh["dp"].get_group(), election="group")
        dist.all_gather_object = counting
        try:
            dcp.load(target, storage_reader=reader, no_dist=True)
        finally:
            dist.all_gather_object = original

        self._assert_equal_sd(sd, target)
        self.assertEqual(
            len(gathered_key_lists),
            1,
            "digests differ, so the key lists must be gathered",
        )

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_validate_flags_an_unsatisfied_request(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        reader = self._reader(path, mesh["dp"].get_group(), validate=True)
        target = self._zeros_like(sd)
        dcp.load(target, storage_reader=reader)  # clean run must not raise
        self._assert_equal_sd(sd, target)

        # Now break the exchange: nothing a rank needs from a peer gets filled,
        # and validate has to notice rather than return a half-loaded state dict.
        # CheckpointException derives from BaseException, so it has to be named:
        # `assertRaises(Exception)` would let it through and kill the process.
        broken = self._reader(path, mesh["dp"].get_group(), validate=True)
        broken._fill = lambda *args, **kwargs: None
        with self.assertRaisesRegex(CheckpointException, "did not satisfy"):
            dcp.load(self._zeros_like(sd), storage_reader=broken)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_load_into_a_different_dtype(self):
        """A load may cast: fp32 on disk into a bf16 model is legal DCP.

        The exchange has to carry the *stored* dtype, which is what every
        receiver sizes its buffer from. Capturing the destination instead would
        broadcast bf16 into fp32 buffers and corrupt or fault; casting belongs
        at each receiver's own destination copy.
        """
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        target = {
            k: DTensor.from_local(
                torch.zeros(v.to_local().shape, dtype=torch.bfloat16, device=v.device),
                device_mesh=v.device_mesh,
                placements=v.placements,
                shape=v.size(),
                stride=v.stride(),
            )
            for k, v in sd.items()
        }
        reader = self._reader(path, mesh["dp"].get_group())
        dcp.load(target, storage_reader=reader)

        self.assertGreater(reader.num_broadcasts, 0)
        # The destination has the wrong dtype for the wire, so the owner had to
        # stage through a stored-dtype scratch rather than broadcast in place.
        self.assertGreater(reader.scratch_bytes, 0)
        for k in sd:
            self.assertEqual(target[k].to_local().dtype, torch.bfloat16)
            self.assertEqual(sd[k].to_local().to(torch.bfloat16), target[k].to_local())

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_world_group_across_tp_shards(self):
        """WORLD as the replication group is correct; the cost is bystanders.

        Ranks holding different TP shards of one FQN want different stored
        chunks, so their keys differ and nothing is deduplicated between them,
        while the dp pairs still share keys and are deduplicated exactly as
        with the dp group. What changes is that every broadcast runs over
        WORLD, so each rank also joins the broadcasts for the other shard,
        allocating a buffer per chunk and discarding it.
        """
        mesh = self._mesh(2, 2)
        sd = self._state_dict(mesh)
        path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(path, thread_count=1))

        stock = self._zeros_like(sd)
        with _count_read_bytes() as counter:
            dcp.load(stock, storage_reader=FileSystemReader(path))
        stock_read = self._total_read(counter["read"])

        tight = self._reader(path, mesh["dp"].get_group())
        target = self._zeros_like(sd)
        with _count_read_bytes() as counter:
            dcp.load(target, storage_reader=tight)
        tight_read = self._total_read(counter["read"])
        self._assert_equal_sd(sd, target)

        wide = self._reader(path, dist.group.WORLD)
        target = self._zeros_like(sd)
        with _count_read_bytes() as counter:
            dcp.load(target, storage_reader=wide)
        wide_read = self._total_read(counter["read"])
        self._assert_equal_sd(sd, target)

        # Same deduplication either way: every stored chunk is read once. The
        # ratio is approximate because every rank also reads the .metadata
        # pickle through the same stream the counter hooks.
        self.assertEqual(wide_read, tight_read)
        self.assertAlmostEqual(stock_read / tight_read, 2, delta=0.05)
        # Twice the broadcasts and twice the bytes on every rank: the other
        # shard's chunks are received and thrown away.
        self.assertEqual(wide.num_broadcasts, 2 * tight.num_broadcasts)
        self.assertEqual(wide.bytes_exchanged, 2 * tight.bytes_exchanged)
        # Contiguous destinations in the stored dtype on the exchange device
        # are broadcast in place: no staging was allocated in either run.
        self.assertEqual(tight.scratch_bytes, 0)
        self.assertEqual(wide.scratch_bytes, 0)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)


class TestReplicaAwareSave(DTensorTestBase):
    """Save-side tests. Needs a CPU backend because async_save requires one."""

    @property
    def world_size(self) -> int:
        return 4

    @property
    def backend(self) -> str:
        return "cpu:gloo,cuda:nccl"

    def _tmpdir(self):
        path = tempfile.mkdtemp(prefix="dcp_repl_save_") if self.rank == 0 else None
        box = [path]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _mesh(self, dp, tp):
        return init_device_mesh(self.device_type, (dp, tp), mesh_dim_names=("dp", "tp"))

    def _state_dict(self, mesh, n=4, dim=256):
        return {
            f"layer{i}.weight": distribute_tensor(
                torch.arange(
                    dim * dim, dtype=torch.float32, device=self.device_type
                ).reshape(dim, dim)
                + i,
                mesh,
                [Replicate(), Shard(0)],
            )
            for i in range(n)
        }

    def _written_mib(self, path):
        import os

        dist.barrier()
        total = 0.0
        if self.rank == 0:
            total = (
                sum(
                    os.path.getsize(os.path.join(path, f))
                    for f in os.listdir(path)
                    if f.endswith(".distcp")
                )
                / 2**20
            )
        box = [total]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _reload_matches(self, sd, path):
        target = {
            k: DTensor.from_local(
                torch.zeros_like(v.to_local()),
                device_mesh=v.device_mesh,
                placements=v.placements,
                shape=v.size(),
                stride=v.stride(),
            )
            for k, v in sd.items()
        }
        dcp.load(target, storage_reader=FileSystemReader(path))
        for k in sd:
            self.assertEqual(sd[k].to_local(), target[k].to_local())

    @with_comms
    def test_writes_the_same_bytes_as_stock(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh)

        stock_path = self._tmpdir()
        dcp.save(sd, storage_writer=FileSystemWriter(stock_path, thread_count=1))
        stock_mib = self._written_mib(stock_path)

        path = self._tmpdir()
        writer = ReplicaAwareStorageWriter(
            FileSystemWriter(path, thread_count=1),
            replication_group=mesh["dp"].get_group(),
        )
        dcp.save(sd, storage_writer=writer)

        self.assertAlmostEqual(self._written_mib(path), stock_mib, delta=0.01)
        # every item was proposed by all dp ranks, so each keeps ~1/dp of them
        self.assertLess(writer.plan_items_out, writer.plan_items_in)
        self._reload_matches(sd, path)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(stock_path, ignore_errors=True)
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_shrinks_the_local_plan(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh, n=8)
        path = self._tmpdir()
        writer = ReplicaAwareStorageWriter(
            FileSystemWriter(path, thread_count=1),
            replication_group=mesh["dp"].get_group(),
        )
        dcp.save(sd, storage_writer=writer)

        kept = [0] * self.world_size
        dist.all_gather_object(kept, writer.plan_items_out)
        self.assertEqual(sum(kept), writer.plan_items_in)  # each item kept exactly once

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_use_collectives_false_raises(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh, n=2)
        path = self._tmpdir()
        writer = ReplicaAwareStorageWriter(
            FileSystemWriter(path, thread_count=1),
            replication_group=mesh["dp"].get_group(),
        )
        # CheckpointException wraps it; the message must name the cause
        with self.assertRaises(BaseException) as caught:
            dcp.save(sd, storage_writer=writer, use_collectives=False)
        self.assertIn("use_collectives=False", str(caught.exception))

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_composes_with_plan_caching(self):
        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh)
        planner = DefaultSavePlanner(enable_plan_caching=True)
        writer_group = mesh["dp"].get_group()

        for _ in range(3):
            path = self._tmpdir()
            writer = ReplicaAwareStorageWriter(
                FileSystemWriter(path, thread_count=1), replication_group=writer_group
            )
            dcp.save(sd, storage_writer=writer, planner=planner)
            self._reload_matches(sd, path)
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_async_save_thread(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh)
        for _ in range(2):
            path = self._tmpdir()
            writer = ReplicaAwareStorageWriter(
                FileSystemWriter(path, thread_count=1),
                replication_group=mesh["dp"].get_group(),
            )
            dcp.async_save(
                sd,
                storage_writer=writer,
                async_checkpointer_type=AsyncCheckpointerType.THREAD,
            ).result()
            self._reload_matches(sd, path)
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_async_save_process_with_rank_partition(self):
        mesh = self._mesh(self.world_size, 1)
        sd = self._state_dict(mesh, n=2)
        partition = [list(range(self.world_size))]
        for _ in range(2):  # the second save must reuse the daemon's cached group
            path = self._tmpdir()
            writer = ReplicaAwareStorageWriter(
                FileSystemWriter(path, thread_count=1), replication_ranks=partition
            )
            dcp.async_save(
                sd,
                storage_writer=writer,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
            ).result()
            self._reload_matches(sd, path)
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_live_process_group_is_not_picklable(self):
        """Why replication_ranks exists at all."""
        import pickle

        mesh = self._mesh(self.world_size, 1)
        path = self._tmpdir()
        with_pg = ReplicaAwareStorageWriter(
            FileSystemWriter(path), replication_group=mesh["dp"].get_group()
        )
        with_ranks = ReplicaAwareStorageWriter(
            FileSystemWriter(path), replication_ranks=[list(range(self.world_size))]
        )
        with self.assertRaises(TypeError):
            pickle.dumps(with_pg)
        pickle.dumps(with_ranks)

        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    run_tests()
