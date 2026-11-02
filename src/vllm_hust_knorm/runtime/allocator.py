"""Shared pool; exclusive request-instance/layer ownership and ABA detection."""

from .state import Page


class PagePool:
    def __init__(self, capacity, reserved=()):
        if capacity < 0 or any(i < 0 or i >= capacity for i in reserved):
            raise ValueError("invalid page pool")
        self.free = set(range(capacity)) - set(reserved)
        self.generations = [0] * capacity
        self.owners = {}
        self.releases = {}

    def allocate(self, owner, count):
        if count < 0:
            raise ValueError("negative allocation")
        if count > len(self.free):
            raise MemoryError("KV page pool exhausted")
        pages = []
        for number in sorted(self.free)[:count]:
            self.free.remove(number)
            self.generations[number] += 1
            page = Page(number, self.generations[number])
            self.owners[page] = owner
            pages.append(page)
        return tuple(pages)

    def validate(self, owner, pages):
        if len(set(pages)) != len(pages):
            raise RuntimeError("duplicate page")
        for page in pages:
            if self.owners.get(page) != owner:
                raise RuntimeError("stale page generation or wrong owner")

    def release(self, owner, pages):
        # Validate the entire batch before touching any ownership.
        self.validate(owner, pages)
        for page in pages:
            del self.owners[page]
            self.free.add(page.number)

    def release_once(self, owner, pages, transaction):
        key = (owner, transaction)
        pages = tuple(pages)
        if key in self.releases:
            if self.releases[key] != pages:
                raise RuntimeError("conflicting release retry")
            return
        self.release(owner, pages)
        self.releases[key] = pages

    def release_request(self, instance):
        for owner in set(self.owners.values()):
            if owner[0] == instance:
                self.release(
                    owner, tuple(p for p, o in self.owners.items() if o == owner)
                )

    def forget_releases(self, instance):
        self.releases = {
            key: pages for key, pages in self.releases.items() if key[0][0] != instance
        }
