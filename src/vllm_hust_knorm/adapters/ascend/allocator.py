"""Generation tracking around the actual vLLM BlockPool, without a second pool."""

from types import MethodType

from ...runtime.state import Layout, Page


class HostAllocator:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.pool = coordinator.block_pool
        if self.pool.enable_caching:
            raise ValueError("prefix caching is incompatible with private layer pages")
        if any(m.block_pool is not self.pool for m in coordinator.single_type_managers):
            raise ValueError("all managers must use the same host BlockPool")
        if hasattr(self.pool, "_knorm_allocator"):
            raise RuntimeError("host pool already has an allocation observer")
        self.pool._knorm_allocator = self
        self.generations = [0] * len(self.pool.blocks)
        self.owners = {}
        self.requests = {}
        self.releases = {}
        original_allocate = self.pool.get_new_blocks

        def allocate(pool, count):
            blocks = original_allocate(count)
            for block in blocks:
                self.generations[block.block_id] += 1
            return blocks

        self.pool.get_new_blocks = MethodType(allocate, self.pool)

    def register(self, request):
        self.requests[request.instance] = request

    def adopt(self, request, lengths):
        self.register(request)
        layouts = {}
        for layer, manager in enumerate(self.coordinator.single_type_managers):
            blocks = manager.req_to_blocks.get(request.request_id, ())
            pages = tuple(
                Page(b.block_id, self.generations[b.block_id]) for b in blocks
            )
            for page in pages:
                block = self.pool.blocks[page.number]
                if block.is_null or block.ref_cnt != 1:
                    raise RuntimeError(
                        "shared or null page in a private request layout"
                    )
                owner = (request.instance, layer)
                if page in self.owners and self.owners[page] != owner:
                    raise RuntimeError("host page already belongs to another layer")
                self.owners[page] = owner
            layouts[layer] = Layout(lengths[layer], pages)
        request.layers = layouts

    def validate(self, owner, pages):
        if len(set(pages)) != len(pages):
            raise RuntimeError("duplicate page")
        for page in pages:
            block = self.pool.blocks[page.number]
            if (
                self.owners.get(page) != owner
                or block.is_null
                or block.ref_cnt != 1
                or self.generations[page.number] != page.generation
            ):
                raise RuntimeError("stale host page or ownership mismatch")

    def install(self, request):
        for layer, layout in request.layers.items():
            self.validate((request.instance, layer), layout.pages)
        for layer, layout in request.layers.items():
            manager = self.coordinator.single_type_managers[layer]
            manager.req_to_blocks[request.request_id] = [
                self.pool.blocks[p.number] for p in layout.pages
            ]

    def release_once(self, owner, pages, transaction):
        key, pages = (owner, transaction), tuple(pages)
        if key in self.releases:
            if self.releases[key] != pages:
                raise RuntimeError("conflicting host release")
            return
        self.validate(owner, pages)
        self.pool.free_blocks([self.pool.blocks[p.number] for p in pages])
        for page in pages:
            del self.owners[page]
        self.releases[key] = pages

    def release_request(self, instance):
        request = self.requests.get(instance)
        if request is None:
            if any(owner[0] == instance for owner in self.owners.values()):
                raise RuntimeError("owned pages without a registered host request")
            return
        for layer, manager in enumerate(self.coordinator.single_type_managers):
            pages = tuple(
                p for p, owner in self.owners.items() if owner == (instance, layer)
            )
            self.validate((instance, layer), pages)
            manager.req_to_blocks.pop(request.request_id, None)
            manager.num_cached_block.pop(request.request_id, None)
            self.pool.free_blocks([self.pool.blocks[p.number] for p in pages])
            for page in pages:
                del self.owners[page]
        del self.requests[instance]

    def forget_releases(self, instance):
        self.releases = {k: v for k, v in self.releases.items() if k[0][0] != instance}
