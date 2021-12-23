import glob
import io
import os
import shutil
import tarfile
import tempfile
import time
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import gt4py as gt
import numpy as np
from gt4py.definitions import FieldInfo
from gt4py.stencil_builder import StencilBuilder
from gt4py.stencil_object import StencilObject

from fv3core.utils.mpi import MPI


class Singleton(type):
    """
    Metaclass that maintains a dictionary of singleton instances.
    """

    _instances: Dict[Type["Singleton"], "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def clear(cls):
        Singleton._instances.clear()


class StencilTable(object, metaclass=Singleton):
    """
    Distributed table to store the status of each stencil based on its ID.
    The status can be one of the following values:

    1. NONE: The stencil ID has not yet been accessed.
    2. DONE: The stencil has been compiled and can be loaded by the calling node.
    3. NODE_ID: The ID of the node that is compiling the requested stencil.

    The table is implemented using one-sided MPI communication (MPI.Window) so
    that nodes can function fully asynchronously. Some nodes may require different
    stencil IDs so it cannot be assumed that all nodes will execute the same code.
    The buffer is of size `max_entries * 2 + 1`, each entry consists of a stencil
    ID and a state, and the first element of the array holds the current size.
    Finished stencil IDs are cached to reduce communication overhead.
    """

    DONE_STATE: int = -1
    NONE_STATE: int = -2
    MAX_SIZE: int = 200
    MAX_STENCIL_BYTES: int = 200000
    NUM_COUNT_BYTES = 2
    NUM_SIZE_BYTES = 4
    NUM_ID_BYTES = 6

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "StencilTable":
        return (
            SequentialTable(*args, **kwargs)
            if MPI is None or MPI.COMM_WORLD.Get_size() == 1
            else DistributedTable(*args, **kwargs)
        )

    @classmethod
    def clear(cls):
        Singleton.clear()

    def set_done(self, key: int) -> None:
        self[key] = self.DONE_STATE
        self._finished_keys.add(key)

    def is_done(self, key: int) -> bool:
        if key in self._finished_keys:
            return True
        if self[key] == self.DONE_STATE:
            self._finished_keys.add(key)
            return True
        return False

    def is_none(self, key: int) -> bool:
        return self[key] == self.NONE_STATE

    def __getitem__(self, key: int) -> int:
        if key in self._finished_keys:
            return self.DONE_STATE

        value: int = self.NONE_STATE
        if key in self._key_nodes:
            node_id, index = self._key_nodes[key]
            buffer = self._get_buffer(node_id)
            assert buffer[index] == key
            value = buffer[index + 1]
        else:
            for node_id in range(self._n_nodes):
                buffer = self._get_buffer(node_id)
                n_items = buffer[0]
                for n in range(n_items):
                    index = n * 2 + 1
                    if buffer[index] == key:
                        value = buffer[index + 1]
                        self._key_nodes[key] = (node_id, index)
                        break

        if value == self.DONE_STATE:
            self._finished_keys.add(key)

        return value

    def __setitem__(self, key: int, value: int) -> None:
        buffer = self._get_buffer()
        if value == self.DONE_STATE:
            self._finished_keys.add(key)

        index: int = -1
        if key in self._key_nodes:
            index = self._key_nodes[key][1]
        else:
            n_items = buffer[0]
            for n in range(n_items):
                pos = n * 2 + 1
                if buffer[pos] == key:
                    index = pos
                    break
            # New entry...
            if index < 0:
                index = n_items * 2 + 1
                buffer[0] = n_items + 1

        buffer[index] = key
        buffer[index + 1] = value
        self._set_buffer(buffer)
        self._key_nodes[key] = (self._node_id, index)

    def _initialize(self, max_size: int = 0):
        max_size = max_size if max_size else self.MAX_SIZE
        self._finished_keys: Set[int] = set()
        self._key_nodes: Dict[int, Tuple[int, int]] = dict()
        self._buffer_size = 2 * max_size + 1
        self._np_type = np.int64
        self._node_id = 0
        self._n_nodes = 1
        self._max_stencil_bytes = self.MAX_STENCIL_BYTES
        self._byte_type = np.byte
        self._stencil_bytes: Optional[np.ndarray] = None

    def _get_stencil_files(self, stencil_object: StencilObject) -> List[str]:
        module_file: str = stencil_object._file_name
        module_prefix: str = module_file.replace(".py", "")
        cache_file: str = f"{module_prefix}.cacheinfo"
        stencil_files = [module_file, cache_file]

        # Add object file (compiled backends)
        object_files: List[str] = glob.glob(f"{module_prefix}_pyext*.so")
        if object_files:
            stencil_files.append(object_files[0])

        # Add computation python file (e.g, gtc:numpy)
        suffix = module_file.split("__")[-1]
        computation_file: str = (
            f"{os.path.dirname(module_prefix)}/m_computation__{suffix}"
        )
        if os.path.exists(computation_file):
            stencil_files.append(computation_file)

        # Add SDFG file (DaCe backends)
        sdfg_file: str = f"{module_prefix}.sdfg"
        if os.path.exists(sdfg_file):
            stencil_files.append(sdfg_file)

        return stencil_files

    def _serialize(self, stencil_object: StencilObject, file_object=None) -> np.ndarray:
        bytes_io = io.BytesIO()
        stencil_files: List[str] = self._get_stencil_files(stencil_object)

        with tarfile.open(fileobj=bytes_io, mode="w:gz") as tar:
            for stencil_file in stencil_files:
                if os.path.isfile(stencil_file):
                    file = open(stencil_file, "rb")
                    file.seek(0, 2)  # go to file end
                    data_len = file.tell()
                    file.seek(0)

                    info = tarfile.TarInfo(stencil_file)
                    info.size = data_len
                    tar.addfile(info, file)
                    file.close()

        bytes_io.seek(0)
        byte_array = bytearray(bytes_io.getvalue())
        np_bytes = np.frombuffer(byte_array, dtype=np.byte)
        if file_object:
            file_object.write(np_bytes)

        return np_bytes

    def _deserialize(self, bytes_array: np.ndarray, extract_dir: str = "") -> None:
        bytes_io = io.BytesIO(bytes_array.tobytes())
        with tarfile.open(fileobj=bytes_io, mode="r:gz") as tar:
            for member in tar.getmembers():
                # TODO(eddied): Disable for now as causing seg faults
                # tar.extract(member)
                if extract_dir:
                    # TODO(eddied): Certainly this code could be more elegant...
                    cache_pos: int = member.path.find(".gt_cache")
                    sub_path: str = member.path[cache_pos + len(".gt_cache") + 1 :]
                    new_path: str = f"{extract_dir}/{sub_path}"
                    if member.path != new_path:
                        dir_name: str = os.path.dirname(new_path)
                        if not os.path.exists(dir_name):
                            os.makedirs(dir_name, exist_ok=True)
                        shutil.copy2(member.path, new_path)

    @abstractmethod
    def _get_buffer(self, node_id: int = 0) -> np.ndarray:
        pass

    @abstractmethod
    def _set_buffer(self, buffer: np.ndarray):
        pass

    def read_stencil(self, stencil_id: int = 0) -> Optional[np.ndarray]:
        if self._stencil_bytes is not None:
            self._deserialize(self._stencil_bytes)
        return self._stencil_bytes

    def write_stencil(self, stencil_class: Type[StencilObject]) -> np.ndarray:
        # Flush previous stencil to gt_cache
        if self._stencil_bytes is not None:
            self._deserialize(self._stencil_bytes, gt.config.cache_settings["dir_name"])
        self._stencil_bytes = self._serialize(stencil_class)
        # TODO(eddied): What if the final stencil is never written to cache?
        return self._stencil_bytes


class SequentialTable(StencilTable):
    def __init__(self, max_size: int = 0):
        """
        Args:
            max_size (int): Maximum number of elements in table
        """
        self._initialize(max_size)

    def _initialize(self, max_size: int = 0):
        super()._initialize(max_size)
        self._window = np.zeros(self._buffer_size, dtype=self._np_type)

    def _get_buffer(self, node_id: int = 0) -> np.ndarray:
        return self._window

    def _set_buffer(self, buffer: np.ndarray):
        self._window = buffer


class DistributedTable(StencilTable):
    def __init__(self, max_size: int = 0, comm: Optional[Any] = None):
        """
        Args:
            max_size (int): Maximum number of elements in table
            comm (Communicator): An MPI communicator (defaults to MPI.COMM_WORLD)
        """
        self._comm = comm if comm else MPI.COMM_WORLD
        self._initialize(max_size)

    def _initialize(self, max_size: int = 0):
        super()._initialize(max_size)

        self._node_id = self._comm.Get_rank()
        self._n_nodes = self._comm.Get_size()

        self._mpi_type = MPI.LONG
        int_size = self._mpi_type.Get_size()
        window_size: int = (
            int_size * self._buffer_size * self._n_nodes if self._node_id == 0 else 0
        )
        self._window = MPI.Win.Allocate(
            size=window_size, disp_unit=int_size, comm=self._comm
        )

        byte_size: int = MPI.BYTE.Get_size()
        window_size = (
            byte_size * self._max_stencil_bytes * self._n_nodes
            if self._node_id == 0
            else 0
        )
        self._byte_window = MPI.Win.Allocate(
            size=window_size, disp_unit=byte_size, comm=self._comm
        )

        if self._node_id == 0:
            # Rank -> Stencil ID mapping table
            buffer = np.frombuffer(self._window, dtype=self._np_type)
            buffer[:] = np.full(len(buffer), self.NONE_STATE, dtype=self._np_type)
            for n in range(self._n_nodes):
                buffer[n * self._buffer_size] = 0

            # Rank -> Stencil Bytes table -- init to zero
            buffer = np.frombuffer(self._byte_window, dtype=np.byte)
            buffer[:] = np.full(len(buffer), 0, dtype=np.byte)

        self._comm.Barrier()

    def _get_buffer(self, node_id: int = -1) -> np.ndarray:
        buffer = np.empty(self._buffer_size, dtype=self._np_type)
        self._window.Lock(rank=0)
        self._window.Get(buffer, target_rank=0, target=self._get_target(node_id))
        self._window.Unlock(rank=0)
        return buffer

    def _set_buffer(self, buffer: np.ndarray) -> None:
        self._window.Lock(rank=0)
        self._window.Put(buffer, target_rank=0, target=self._get_target())
        self._window.Unlock(rank=0)

    def _get_target(self, node_id: int = -1) -> Tuple[int, int, Any]:
        if node_id < 0:
            node_id = self._node_id
        return (node_id * self._buffer_size, self._buffer_size, self._mpi_type)

    def _read_byte_window(self) -> np.ndarray:
        buffer_size: int = self._max_stencil_bytes * self._n_nodes
        rank: int = 0
        target = (rank, buffer_size, MPI.BYTE)
        buffer: np.ndarray = np.empty(buffer_size, dtype=np.byte)

        self._byte_window.Lock(rank=rank)
        self._byte_window.Get(buffer, target_rank=rank, target=target)
        self._byte_window.Unlock(rank=rank)

        return buffer

    def _write_byte_window(self, buffer: np.ndarray) -> None:
        buffer_size: int = self._max_stencil_bytes * self._n_nodes
        rank: int = 0
        target = (rank, buffer_size, MPI.BYTE)

        self._byte_window.Lock(rank=rank)
        self._byte_window.Put(buffer, target_rank=rank, target=target)
        self._byte_window.Unlock(rank=rank)

    def read_stencil(self, stencil_id: int = 0) -> Optional[np.ndarray]:
        # Read bytes from window
        buffer = self._read_byte_window()

        endian: str = "big"
        count_bytes: bytes = buffer[0 : self.NUM_COUNT_BYTES].tobytes()
        n_stencils: int = int.from_bytes(count_bytes, endian)
        offset: int = self.NUM_COUNT_BYTES
        n_stencil_bytes: int = 0

        # Find the stencil with the matching stencil ID
        stencil_found: bool = False
        for _ in range(n_stencils):
            # Next six bytes store the stencil ID
            curr_stencil_id: int = int.from_bytes(
                buffer[offset : offset + self.NUM_ID_BYTES].tobytes(), endian
            )
            offset += self.NUM_ID_BYTES

            # Next four bytes store the size of the serialized stencil
            n_stencil_bytes = int.from_bytes(
                buffer[offset : offset + self.NUM_SIZE_BYTES].tobytes(), endian
            )
            offset += self.NUM_SIZE_BYTES

            stencil_found = curr_stencil_id == stencil_id
            if stencil_found:
                break

            # Increase offset by size bytes
            offset += n_stencil_bytes

        if not stencil_found:
            raise RuntimeError(
                f"Stencil ID {stencil_id} not found in distributed table"
            )

        self._stencil_bytes = buffer[offset : offset + n_stencil_bytes]

        with open(f"./future_stencil_r{MPI.COMM_WORLD.Get_rank()}.log", "a") as log:
            log.write(
                f"{time.time()} [read_stencil]: stencil_id = {stencil_id}, n_stencil_bytes = {n_stencil_bytes}, len(buffer) = {buffer.size}, offset = {offset}, buffer[offset + 2:stencil_bytes.size + offset + 2].size = {buffer[offset + 2:self._stencil_bytes.size + offset + 2].size}, stencil_bytes.size = {self._stencil_bytes.size}\n"
            )

        return super().read_stencil(stencil_id)

    def write_stencil(self, stencil_class: Type[StencilObject]) -> np.ndarray:
        # Serialize the stencil
        stencil_bytes = super().write_stencil(stencil_class)

        # Read bytes from window
        buffer = self._read_byte_window()

        # First two bytes store the number of stencils
        endian: str = "big"
        n_stencils: int = int.from_bytes(
            buffer[0 : self.NUM_COUNT_BYTES].tobytes(), endian
        )
        offset: int = self.NUM_COUNT_BYTES

        # Find the next available offset location
        for _ in range(n_stencils):
            # Skip stencil ID bytes
            offset += self.NUM_ID_BYTES

            # Next four bytes store the size of the serialized stencil
            n_stencil_bytes: int = int.from_bytes(
                buffer[offset : offset + self.NUM_SIZE_BYTES].tobytes(), endian
            )

            # Increase offset by size bytes
            offset += self.NUM_SIZE_BYTES + n_stencil_bytes

        # Write the stencil ID
        stencil_id = int(stencil_class._gt_id_, 16)
        id_bytes: bytes = stencil_id.to_bytes(self.NUM_ID_BYTES, endian)
        buffer[offset : offset + self.NUM_ID_BYTES] = list(id_bytes)
        offset += self.NUM_ID_BYTES

        # Write the size of this stencil
        size_bytes: bytes = stencil_bytes.size.to_bytes(self.NUM_SIZE_BYTES, endian)
        buffer[offset : offset + self.NUM_SIZE_BYTES] = list(size_bytes)
        offset += self.NUM_SIZE_BYTES

        # Write the stencil bytes
        buffer[offset : offset + stencil_bytes.size] = stencil_bytes

        # Increment the stencil count
        n_stencils += 1
        count_bytes: bytes = n_stencils.to_bytes(2, endian)
        buffer[0 : self.NUM_COUNT_BYTES] = list(count_bytes)

        # Write the buffer to the one-sided byte window
        self._write_byte_window(buffer)

        with open(f"./future_stencil_r{MPI.COMM_WORLD.Get_rank()}.log", "a") as log:
            log.write(
                f"{time.time()} [write_stencil]: stencil_id = {stencil_id}, size_bytes = {size_bytes}, len(buffer) = {buffer.size}, offset = {offset}, buffer[offset + 2:stencil_bytes.size + offset + 2].size = {buffer[offset + 2:stencil_bytes.size + offset + 2].size}, stencil_bytes.size = {stencil_bytes.size}, n_stencils = {n_stencils}\n"
            )

        return stencil_bytes


def future_stencil(
    backend: Optional[str] = None,
    definition: Optional[Callable] = None,
    *,
    externals: Optional[Dict[str, Any]] = None,
    wrapper: Optional[Callable] = None,
    rebuild: bool = False,
    **kwargs: Any,
):
    """
    Create a future stencil object with deferred building in a distributed context

    Parameters
    ----------
        backend : `str`
            Name of the implementation backend.

        definition : `None` when used as a decorator, otherwise a `function` or a
                     `:class:`gt4py.StencilObject`
            Function object defining the stencil.

        externals: `dict`, optional
            Specify values for otherwise unbound symbols.

        rebuild : `bool`, optional
            Force rebuild of the :class:`gt4py.StencilObject` even if it is
            found in the cache. (`False` by default).

        **kwargs: `dict`, optional
            Extra backend-specific options. Check the specific backend
            documentation for further information.

    Returns
    -------
        :class:`FutureStencil`
            Wrapper around an instance of the dynamically-generated subclass
            of :class:`gt4py.StencilObject`.
    """

    def _decorator(func):
        # Move backend options to `backend_opts`
        backend_opts: Dict[str, Any] = {}
        for backend_opt in ("device_sync", "skip_passes", "verbose"):
            if backend_opt in kwargs:
                backend_opts[backend_opt] = kwargs.pop(backend_opt)

        builder = (
            StencilBuilder(func)
            .with_backend(backend)
            .with_externals(externals or {})
            .with_options(
                name=func.__name__,
                module=func.__module__,
                rebuild=rebuild,
                backend_opts=backend_opts,
                **kwargs,
            )
        )
        return FutureStencil(builder, wrapper)

    if definition is None:
        return _decorator
    return _decorator(definition)


class FutureStencil:
    """
    A wrapper that allows a stencil object to be compiled in a distributed context.
    """

    def __init__(
        self,
        builder: Optional["StencilBuilder"] = None,
        wrapper: Optional[Callable] = None,
        sleep_time: float = 0.05,
        timeout: float = 600.0,
    ):
        """
        Args:
            builder: StencilBuilder object to build the stencil
            wrapper: Another wrapper with a `stencil_object` attribute to which the
                     compiled stencil can be passed
            sleep_time: Amount of time to sleep between table checks (defaults to 50 ms)
            timeout: Time to wait for a stencil to compile (defaults to 10 min)
        """
        self._builder = builder
        self._wrapper = wrapper
        self._sleep_time = sleep_time
        self._timeout = timeout
        self._id_table = StencilTable.create()
        self._node_id: int = MPI.COMM_WORLD.Get_rank() if MPI else 0
        self._stencil_object: Optional[StencilObject] = None

    @property
    def cache_info_path(self) -> str:
        return self._builder.caching.cache_info_path.stem

    @property
    def stencil_object(self) -> StencilObject:
        if self._stencil_object is None:
            self._wait_for_stencil()
        return self._stencil_object

    @property
    def field_info(self) -> Dict[str, FieldInfo]:
        return self.stencil_object.field_info

    @property
    def stencil_table(self) -> StencilTable:
        return self._id_table

    def _delay(self, factor: float = 0.4) -> float:
        delay_time = self._sleep_time * float(self._node_id) * factor
        time.sleep(delay_time)
        return delay_time

    def _compile_stencil(self, stencil_id: int) -> Callable:
        # Stencil not yet compiled or in progress so claim it...
        self._id_table[stencil_id] = self._node_id
        self._delay()

        stencil_class = self._builder.backend.generate()
        self._id_table.write_stencil(stencil_class)
        self._id_table.set_done(stencil_id)

        return stencil_class

    def _load_stencil(self, stencil_id: int) -> Callable:
        if not self._id_table.is_done(stencil_id):
            # Wait for stencil to be done...
            time_elapsed: float = 0.0
            while (
                not self._id_table.is_done(stencil_id) and time_elapsed < self._timeout
            ):
                time_elapsed += self._delay()

            if time_elapsed >= self._timeout:
                raise RuntimeError(
                    f"Timeout while waiting for stencil '{self.cache_info_path}' "
                    "to compile on node {self._node_id}"
                )

        # Delay before loading...
        self._delay()
        self._id_table.read_stencil(stencil_id)
        stencil_class = self._builder.backend.load()

        return stencil_class

    def _load_cached_stencil(self):
        stencil_class: Callable = None
        self._delay()

        # try/except block to prevent loading incomplete files, either
        #  1. Attribute errors due to missing 'run' or 'call' methods
        #  2. The gt4py caching system tries to create existing directory
        #  3. File not found errors if an expected file does not yet exist
        try:
            stencil_class = self._builder.backend.load()
        except (AttributeError, FileExistsError, FileNotFoundError):
            stencil_class = None

        return stencil_class

    def _wait_for_stencil(self):
        builder = self._builder
        stencil_id = int(builder.stencil_id.version, 16)

        stencil_class: Callable = None
        if not builder.options.rebuild:
            stencil_class = self._load_cached_stencil()

        # Redirect cache to temporary directory
        gt_cache_dir_name: str = gt.config.cache_settings["dir_name"]
        gt.config.cache_settings["dir_name"] = f"{tempfile.gettempdir()}/.gt_cache"

        if not stencil_class:
            # Delay before accessing distributed table...
            self._delay()
            if self._id_table.is_none(stencil_id):
                # Compile stencil and mark as DONE...
                stencil_class = self._compile_stencil(stencil_id)
            else:
                # Wait for stencil and load from cache...
                stencil_class = self._load_stencil(stencil_id)

        # Restore cache location
        gt.config.cache_settings["dir_name"] = gt_cache_dir_name

        if not stencil_class:
            raise RuntimeError(
                f"`stencil_class` is None '{self.cache_info_path}' ({stencil_id})!"
            )

        self._stencil_object = stencil_class()

        # Assign wrapper's stencil_object (e.g,. FrozenStencil) if provided...
        if self._wrapper:
            self._wrapper.stencil_object = self._stencil_object

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self.stencil_object(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.stencil_object.run(*args, **kwargs)
